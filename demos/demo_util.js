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
import * as tf from '@tensorflow/tfjs';

const color = 'aqua';
const boundingBoxColor = 'red';
const lineWidth = 2;

function toTuple({y, x}) {
  return [y, x];
}

export function drawPoint(ctx, y, x, r, color) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}

/**
 * Draws a line on a canvas, i.e. a joint
 */
export function drawSegment([ay, ax], [by, bx], color, scale, ctx) {
  ctx.beginPath();
  ctx.moveTo(ax * scale, ay * scale);
  ctx.lineTo(bx * scale, by * scale);
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = color;
  ctx.stroke();
}

/**
 * Draws a pose skeleton by looking up all adjacent keypoints/joints
 */
export function drawSkeleton(keypoints, minConfidence, ctx, scale = 1) {
  const adjacentKeyPoints =
      posenet.getAdjacentKeyPoints(keypoints, minConfidence);

  adjacentKeyPoints.forEach((keypoints) => {
    drawSegment(
        toTuple(keypoints[0].position), toTuple(keypoints[1].position), color,
        scale, ctx);
  });
}

/**
 * Draw pose keypoints onto a canvas
 */
export function drawKeypoints(keypoints, minConfidence, ctx, scale = 1) {
  for (let i = 0; i < keypoints.length; i++) {
    const keypoint = keypoints[i];

    if (keypoint.score < minConfidence) {
      continue;
    }

    const {y, x} = keypoint.position;
    drawPoint(ctx, y * scale, x * scale, 3, color);
  }
}

/**
 * Draw the bounding box of a pose. For example, for a whole person standing
 * in an image, the bounding box will begin at the nose and extend to one of
 * ankles
 */
export function drawBoundingBox(keypoints, ctx) {
  const boundingBox = posenet.getBoundingBox(keypoints);

  ctx.rect(
      boundingBox.minX, boundingBox.minY, boundingBox.maxX - boundingBox.minX,
      boundingBox.maxY - boundingBox.minY);

  ctx.strokeStyle = boundingBoxColor;
  ctx.stroke();
}

export function getBoundingBoxSize(keypoints) {
  const boundingBox = posenet.getBoundingBox(keypoints);
  return (boundingBox.maxX - boundingBox.minX) * (boundingBox.maxY - boundingBox.minY);
}

/**
 * Converts an arary of pixel data into an ImageData object
 */
export async function renderToCanvas(a, ctx) {
  const [height, width] = a.shape;
  const imageData = new ImageData(width, height);

  const data = await a.data();

  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    const k = i * 3;

    imageData.data[j + 0] = data[k + 0];
    imageData.data[j + 1] = data[k + 1];
    imageData.data[j + 2] = data[k + 2];
    imageData.data[j + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
}

/**
 * Draw an image on a canvas
 */
export function renderImageToCanvas(image, size, canvas) {
  canvas.width = size[0];
  canvas.height = size[1];
  const ctx = canvas.getContext('2d');

  ctx.drawImage(image, 0, 0);
}

/**
 * Draw heatmap values, one of the model outputs, on to the canvas
 * Read our blog post for a description of PoseNet's heatmap outputs
 * https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5
 */
export function drawHeatMapValues(heatMapValues, outputStride, canvas) {
  const ctx = canvas.getContext('2d');
  const radius = 5;
  const scaledValues = heatMapValues.mul(tf.scalar(outputStride, 'int32'));

  drawPoints(ctx, scaledValues, radius, color);
}

/**
 * Used by the drawHeatMapValues method to draw heatmap points on to
 * the canvas
 */
function drawPoints(ctx, points, radius, color) {
  const data = points.buffer().values;

  for (let i = 0; i < data.length; i += 2) {
    const pointY = data[i];
    const pointX = data[i + 1];

    if (pointX !== 0 && pointY !== 0) {
      ctx.beginPath();
      ctx.arc(pointX, pointY, radius, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
    }
  }
}

/**
 * Draw offset vector values, one of the model outputs, on to the canvas
 * Read our blog post for a description of PoseNet's offset vector outputs
 * https://medium.com/tensorflow/real-time-human-pose-estimation-in-the-browser-with-tensorflow-js-7dd0bc881cd5
 */
export function drawOffsetVectors(
    heatMapValues, offsets, outputStride, scale = 1, ctx) {
  const offsetPoints =
      posenet.singlePose.getOffsetPoints(heatMapValues, outputStride, offsets);

  const heatmapData = heatMapValues.buffer().values;
  const offsetPointsData = offsetPoints.buffer().values;

  for (let i = 0; i < heatmapData.length; i += 2) {
    const heatmapY = heatmapData[i] * outputStride;
    const heatmapX = heatmapData[i + 1] * outputStride;
    const offsetPointY = offsetPointsData[i];
    const offsetPointX = offsetPointsData[i + 1];

    drawSegment(
        [heatmapY, heatmapX], [offsetPointY, offsetPointX], color, scale, ctx);
  }
}

export function existsArms(keypoints, minConfidence) {
  // 5: leftShoulder, 6: rightShoulder
  // 7: leftElbow, 8: rightElbow
  // 9: leftWrist, 10: rightWrist
  if ((keypoints[5].score > minConfidence) &&
      (keypoints[6].score > minConfidence) &&
      (keypoints[7].score > minConfidence) &&
      (keypoints[8].score > minConfidence))
    return {left: true, right: true};
  else if ((keypoints[5].score > minConfidence) &&
           (keypoints[6].score > minConfidence) &&
           (keypoints[7].score > minConfidence))
    return {left: true, right: false};
  else if ((keypoints[5].score > minConfidence) &&
           (keypoints[6].score > minConfidence) &&
           (keypoints[8].score > minConfidence))
    return {left: false, right: true}
  else
    return {left: false, right: false}
}

export function existWritst(keypoints, minConfidence) {
  if ((keypoints[9].score > minConfidence) &&
       keypoints[10].score > minConfidence)
    return {left: true, right: true};
  else if (keypoints[9].score > minConfidence)
    return {left: true, right: false};
  else if (keypoints[10].score > minConfidence)
    return {left: false, right: true};
  else
    return {left: false, right: false};
}

export function existsEyeAndNose(keypoints, minConfidence) {
    // 0	nose
    // 1	leftEye
    // 2	rightEye
    // 3	leftEar
    // 4	rightEar
    if ((keypoints[0].score > minConfidence) &&
        (keypoints[1].score > minConfidence) &&
        (keypoints[2].score > minConfidence))
      return true;
    else
      return false;
}

export function existsNoseAndEyeAndEar(keypoints, minConfidence) {
  if ((keypoints[0].score > minConfidence) &&
      (keypoints[1].score > minConfidence) &&
      (keypoints[2].score > minConfidence))
    if ((keypoints[3].score > minConfidence) &&
        (keypoints[4].score > minConfidence))
      return {left: true, right: true};
    else if (keypoints[3].score > minConfidence)
      return {left: true, right: false};
    else if (keypoints[4].score > minConfidence)
      return {left: false, right: true};
    else
      return {left: false, right: false};
  else
    return {left: false, right: false};
}

export function getFaceYaw(keypoints) {
  // 両目と鼻による三角形において鼻から両目の線分に垂線を下ろした点Pが
  // 両目の線分のどの辺りに位置するか計算
  const nose = keypoints[0].position;
  const leftEye = keypoints[1].position;
  const rightEye = keypoints[2].position;
  var eyeVector = {};
  eyeVector.x = rightEye.x - leftEye.x;
  eyeVector.y = rightEye.y - leftEye.y;
  var eyeToNoseVector = {};
  // TODO: これでよい？
  eyeToNoseVector.x = nose.x - leftEye.x;
  eyeToNoseVector.y = eyeVector.y;
  var ratio = norm(eyeToNoseVector) / norm(eyeVector);
  return 180 - ratio * 180;
}

export function getFacePitch(keypoints, earFlag) {
  // 
  const leftEye = keypoints[1].position;
  const rightEye = keypoints[2].position;
  const nose = keypoints[0].position;
  var eyeVector = {};
  eyeVector.x = rightEye.x - leftEye.x;
  eyeVector.y = rightEye.y - leftEye.y;
  var eyeToNoseVector = {};
  eyeToNoseVector = {};
  eyeToNoseVector.x = nose.x - leftEye.x;
  eyeToNoseVector.y = nose.y - leftEye.y;
  var angle = calculateAngle(eyeToNoseVector, eyeVector);
  var triangleHeight = norm(eyeToNoseVector) * Math.sin(angle * (Math.PI / 180));
  if (earFlag.left && earFlag.right) {
    const leftEar = keypoints[3].position;
    const rightEar = keypoints[4].position;
    var earVector = {};
    earVector.x = rightEar.x - leftEar.x;
    earVector.y = rightEar.y - leftEar.y;
    var earToNoseVector = {};
    earToNoseVector.x = nose.x - leftEar.x;
    earToNoseVector.y = nose.y - leftEar.y;
    var angle = calculateAngle(earToNoseVector, earVector);
    var noseHeight = norm(earToNoseVector) * Math.sin(angle * (Math.PI / 180));
  } else {
    var idx = earFlag.left ? 3 : 4;
    const ear = keypoints[idx].position;
    var earVector = {};
    earVector.x = ear.x + eyeVector.x;
    earVector.y = ear.y + eyeVector.y;
    // TODO
  }
}

function getShoulderLine(keypoints) {
  const leftShoulder = keypoints[5].position;
  const rightShoulder = keypoints[6].position;
  let x = leftShoulder.x - rightShoulder.x;
  let y = leftShoulder.y - rightShoulder.y;
  return {x: x, y: y};
}

function getUpperArmLine(keypoints) {
  const leftShoulder = keypoints[5].position;
  const rightShoulder = keypoints[6].position;
  const leftElbow = keypoints[7].position;
  const rightElbow = keypoints[8].position;
  let left_x = leftElbow.x - leftShoulder.x;
  let left_y = leftElbow.y - leftShoulder.y;
  let right_x = rightElbow.x - rightShoulder.x;
  let right_y = rightElbow.y - rightShoulder.y;
  return {
    left: {
      x: left_x,
      y: left_y
    },
    right: {
      x: right_x,
      y: right_y
    }
  }
}

export function getArmsAngle(keypoints) {
  let shoulderLine = getShoulderLine(keypoints);
  let upperArmLine = getUpperArmLine(keypoints);
  return {
    left: calculateAngle(rotateVector(shoulderLine, 90), upperArmLine.left),
    right: calculateAngle(rotateVector(shoulderLine, 90), upperArmLine.right)
  }
}

function getWristLine(keypoints) {
  const leftShoulder = keypoints[5].position;
  const rightShoulder = keypoints[6].position;
  const leftWrist = keypoints[9].position;
  const rightWrist = keypoints[10].position;
  let left_x = leftWrist.x - leftShoulder.x;
  let left_y = leftWrist.y - leftShoulder.y;
  let right_x = rightWrist.x - rightShoulder.x;
  let right_y = rightWrist.y - rightShoulder.y;
  return {
    left: {
      x: left_x,
      y: left_y
    },
    right: {
      x: right_x,
      y: right_y
    }
  }
}

export function getWristAngle(keypoints) {
  let shoulderLine = getShoulderLine(keypoints);
  let wristLine = getWristLine(keypoints);
  return {
    left: calculateAngle(rotateVector(shoulderLine, 90), wristLine.left),
    right: calculateAngle(rotateVector(shoulderLine, 90), wristLine.right)
  }
}

function rotateVector(vector, degree) {
  const rad = degree * (Math.PI / 180);
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  return {
    x: vector.x * cos - vector.y * sin,
    y: vector.x * sin + vector.y * cos
  }
}

function calculateAngle(vectorA, vectorB) {
  const dot = vectorA.x * vectorB.x + vectorA.y * vectorB.y;
  const cos = dot / (norm(vectorA) * norm(vectorB));
  const rad = Math.acos(cos);
  return rad * 180 / Math.PI;
}

function norm(vector) {
  return Math.sqrt(vector.x * vector.x + vector.y * vector.y);
}