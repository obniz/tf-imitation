/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as posenet from '@tensorflow-models/posenet';
import dat from 'dat.gui';
import Stats from 'stats.js';

import {drawBoundingBox, drawKeypoints, drawSkeleton, getBoundingBoxSize,
        existsArms, getShoulderLine, getUpperArmLine, getArmsAngle,
        getWristAngle, getFaceYaw, existsEyeAndNose} from './demo_util';
import { start } from 'repl';
import { math } from '@tensorflow/tfjs';

const videoWidth = 600;
const videoHeight = 500;
const stats = new Stats();

var left_angles = [];
var right_angles = [];
var faceYaws = [];
var facePitchs = [];

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}

/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined : videoHeight,
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}

const guiState = {
  algorithm: 'multi-pose',
  input: {
    mobileNetArchitecture: isMobile() ? '0.50' : '0.75',
    outputStride: 16,
    imageScaleFactor: 0.5,
  },
  singlePoseDetection: {
    minPoseConfidence: 0.1,
    minPartConfidence: 0.5,
  },
  multiPoseDetection: {
    maxPoseDetections: 5,
    minPoseConfidence: 0.15,
    minPartConfidence: 0.1,
    nmsRadius: 30.0,
  },
  output: {
    showVideo: true,
    showSkeleton: true,
    showPoints: true,
    showBoundingBox: false,
    showAll: false,
  },
  net: null,
};

/**
 * Sets up dat.gui controller on the top-right of the window
 */
function setupGui(cameras, net) {
  guiState.net = net;

  if (cameras.length > 0) {
    guiState.camera = cameras[0].deviceId;
  }

  const gui = new dat.GUI({width: 300});

  // The single-pose algorithm is faster and simpler but requires only one
  // person to be in the frame or results will be innaccurate. Multi-pose works
  // for more than 1 person
  const algorithmController =
      gui.add(guiState, 'algorithm', ['single-pose', 'multi-pose']);

  // The input parameters have the most effect on accuracy and speed of the
  // network
  let input = gui.addFolder('Input');
  // Architecture: there are a few PoseNet models varying in size and
  // accuracy. 1.01 is the largest, but will be the slowest. 0.50 is the
  // fastest, but least accurate.
  const architectureController = input.add(
      guiState.input, 'mobileNetArchitecture',
      ['1.01', '1.00', '0.75', '0.50']);
  // Output stride:  Internally, this parameter affects the height and width of
  // the layers in the neural network. The lower the value of the output stride
  // the higher the accuracy but slower the speed, the higher the value the
  // faster the speed but lower the accuracy.
  input.add(guiState.input, 'outputStride', [8, 16, 32]);
  // Image scale factor: What to scale the image by before feeding it through
  // the network.
  input.add(guiState.input, 'imageScaleFactor').min(0.2).max(1.0);
  input.open();

  // Pose confidence: the overall confidence in the estimation of a person's
  // pose (i.e. a person detected in a frame)
  // Min part confidence: the confidence that a particular estimated keypoint
  // position is accurate (i.e. the elbow's position)
  let single = gui.addFolder('Single Pose Detection');
  single.add(guiState.singlePoseDetection, 'minPoseConfidence', 0.0, 1.0);
  single.add(guiState.singlePoseDetection, 'minPartConfidence', 0.0, 1.0);

  let multi = gui.addFolder('Multi Pose Detection');
  multi.add(guiState.multiPoseDetection, 'maxPoseDetections')
      .min(1)
      .max(20)
      .step(1);
  multi.add(guiState.multiPoseDetection, 'minPoseConfidence', 0.0, 1.0);
  multi.add(guiState.multiPoseDetection, 'minPartConfidence', 0.0, 1.0);
  // nms Radius: controls the minimum distance between poses that are returned
  // defaults to 20, which is probably fine for most use cases
  multi.add(guiState.multiPoseDetection, 'nmsRadius').min(0.0).max(40.0);
  multi.open();

  let output = gui.addFolder('Output');
  output.add(guiState.output, 'showVideo');
  output.add(guiState.output, 'showSkeleton');
  output.add(guiState.output, 'showPoints');
  output.add(guiState.output, 'showBoundingBox');
  output.add(guiState.output, 'showAll');
  output.open();


  architectureController.onChange(function(architecture) {
    guiState.changeToArchitecture = architecture;
  });

  algorithmController.onChange(function(value) {
    switch (guiState.algorithm) {
      case 'single-pose':
        multi.close();
        single.open();
        break;
      case 'multi-pose':
        single.close();
        multi.open();
        break;
    } 
  });
}

/**
 * Sets up a frames per second panel on the top-left of the window
 */
function setupFPS() {
  stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild(stats.dom);
}

/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(video, net) {
  const canvas = document.getElementById('output');
  const ctx = canvas.getContext('2d');
  // since images are being fed from a webcam
  const flipHorizontal = true;

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  async function poseDetectionFrame() {
    if (guiState.changeToArchitecture) {
      // Important to purge variables and free up GPU memory
      guiState.net.dispose();

      // Load the PoseNet model weights for either the 0.50, 0.75, 1.00, or 1.01
      // version
      guiState.net = await posenet.load(+guiState.changeToArchitecture);

      guiState.changeToArchitecture = null;
    }

    // Begin monitoring code for frames per second
    stats.begin();

    // Scale an image down to a certain factor. Too large of an image will slow
    // down the GPU
    const imageScaleFactor = guiState.input.imageScaleFactor;
    const outputStride = +guiState.input.outputStride;

    let poses = [];
    let minPoseConfidence;
    let minPartConfidence;
    switch (guiState.algorithm) {
      case 'single-pose':
        const pose = await guiState.net.estimateSinglePose(
            video, imageScaleFactor, flipHorizontal, outputStride);
        poses.push(pose);
        minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
        break;
      case 'multi-pose':
        poses = await guiState.net.estimateMultiplePoses(
            video, imageScaleFactor, flipHorizontal, outputStride,
            guiState.multiPoseDetection.maxPoseDetections,
            guiState.multiPoseDetection.minPartConfidence,
            guiState.multiPoseDetection.nmsRadius);
        minPoseConfidence = +guiState.multiPoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.multiPoseDetection.minPartConfidence;
        break;
    }

    ctx.clearRect(0, 0, videoWidth, videoHeight);

    if (guiState.output.showVideo) {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-videoWidth, 0);
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      ctx.restore();
    }

    // For each pose (i.e. person) detected in an image, loop through the poses
    // and draw the resulting skeleton and keypoints if over certain confidence
    // scores
    // 確度の高いもののみのリストを作成
    let confidentPoses = []
    poses.forEach((elm) => {
      if (elm.score >= minPoseConfidence) {
        confidentPoses.push(elm);
      }
    }, this);
    if (guiState.output.showAll) {
      confidentPoses.forEach(({_, keypoints}) => {
        if (guiState.output.showPoints) {
          drawKeypoints(keypoints, minPartConfidence, ctx);
        }
        if (guiState.output.showSkeleton) {
          drawSkeleton(keypoints, minPartConfidence, ctx);
        }
        if (guiState.output.showBoundingBox) {
          drawBoundingBox(keypoints, ctx);
        }
      });
    }
    // 一番大きい人物を取得
    let maxSize = 0;
    let maxPose;
    for (let pose of confidentPoses) {
      let size = getBoundingBoxSize(pose.keypoints);
      if (size > maxSize) {
        maxSize = size;
        maxPose = pose;
      }
    }
    if (maxPose) {
      if (!guiState.output.showAll) {
        if (guiState.output.showPoints)
          drawKeypoints(maxPose.keypoints, minPartConfidence, ctx);
        if (guiState.output.showSkeleton)
          drawSkeleton(maxPose.keypoints, minPartConfidence, ctx);
        if (guiState.output.showBoundingBox)
          drawBoundingBox(maxPose.keypoints, ctx);
      }
      
      // 一番大きい人物の腕取得
      const offset = 30;
      if (existsArms(maxPose.keypoints, minPartConfidence).left) {
        let angles = getArmsAngle(maxPose.keypoints);
        let wristAngles = getWristAngle(maxPose.keypoints);
        if (angles.left < wristAngles.left)
          left_angles.push(wristAngles.left + offset);
        else
          left_angles.push(angles.left + offset);
      }
      if (existsArms(maxPose.keypoints, minPartConfidence).right) {
        let angles = getArmsAngle(maxPose.keypoints);
        let wristAngles = getWristAngle(maxPose.keypoints);
        if (angles.right < wristAngles.right)
          right_angles.push(180 - wristAngles.right - offset);
        else
          right_angles.push(180 - angles.right - offset);
      }
      // 一番大きい人物の顔取得
      // yaw軸
      if (existsEyeAndNose(maxPose.keypoints, minPartConfidence)) {
        let faceYaw = getFaceYaw(maxPose.keypoints);
        faceYaws.push(faceYaw);
      }
    }

    // End monitoring code for frames per second
    stats.end();

    requestAnimationFrame(poseDetectionFrame);
  }

  poseDetectionFrame();
}

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime function.
 */
export async function bindPage() {
  // Load the PoseNet model weights with architecture 0.75
  const net = await posenet.load(0.75);
  
  let obniz = new Obniz("5976-2157");
  obniz.onconnect = async function () {
    let rightServo = obniz.wired("ServoMotor", {signal: 0, vcc: 1, gnd: 2});
    let leftServo = obniz.wired("ServoMotor", {signal: 3, vcc: 4, gnd: 5});
    let faceYawServo = obniz.wired("ServoMotor", {signal: 6, vcc: 7, gnd: 8});

    rightServo.status = 150;
    rightServo.elmId = "right";
    rightServo.angle(rightServo.status);
    leftServo.status = 30;
    leftServo.elmId = "left";
    leftServo.angle(leftServo.status);
    faceYawServo.status = 90;
    faceYawServo.angle(faceYawServo.status);

    startServo(rightServo, right_angles, 200);
    startServo(leftServo, left_angles, 200);
    startServo(faceYawServo, faceYaws, 200);
    
  }
  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'block';

  let video;

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
        'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  setupGui([], net);
  setupFPS();
  detectPoseInRealTime(video, net);
}

async function moveServoTo(servo, to) {
  if (to < 0 || to > 180)
    return;
  if (servo.status < to) 
    for (let i = servo.status; i <= to; i++) {
      servo.angle(i);
      servo.status = i;
      if (servo.elmId != undefined)
        document.getElementById(servo.elmId).innerHTML = servo.status;
      await servo.obniz.wait(5);
    }
  else if (servo.status > to)
    for (let i = servo.status; i >= to; i--) {
      servo.angle(i);
      servo.status = i;
      if (servo.elmId != undefined)
        document.getElementById(servo.elmId).innerHTML = servo.status;
      await servo.obniz.wait(5);
    }
}

function startServo(servo, angles, interval) {
  let updateServo = setInterval( () => {
    if (angles.length > 0) {
      let med = median(angles);
      if (Math.abs(servo.status - med) > 5) {
        // moveServoTo(servo, med);
        servo.angle(med)
      }
      angles.splice(0);
    }
  }, interval);
}

function median(list) {
  let half = (list.length / 2) | 0;
  let sorted = list.slice();
  sorted.sort();
  if (sorted.length % 2) {
    return sorted[half];
  }

  return (sorted[half - 1] + sorted[half]) / 2;
}

function sleep(a){
  var dt1 = new Date().getTime();
  var dt2 = new Date().getTime();
  while (dt2 < dt1 + a){
    dt2 = new Date().getTime();
  }
  return;
}
navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();
