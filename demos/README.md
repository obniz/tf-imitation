# PoseNet Demos

## Description
### 接続
tensorflow/tf-models を用いて作成しました．
camera.js がメインのソースコードで，`bindPage()`内でページを開いた時のモデル読み込みからonbiz準備までを行なっています．
obnizの接続は，
- 右手サーボ(ロボットの)：signal:0，vcc:1，gnd:2
- 左手サーボ(ロボットの)：signal:3，vcc:4，gnd:5
- 頭サーボ：signal:6，vcc:7，gnd:8
としてください．

### 起動
クローン後`npm install`を行なってください．  
ソースコードを更新した場合`npm run build`を行なってください．  
`dist/camera.html`でデモページがひらけます．  
認識部のパラメータ調整やGUI上に表示するものなどは画面右側のGUIで操作できます．  


## Contents

### Demo 1: Camera

The camera demo shows how to estimate poses in real-time from a webcam video stream.

<img src="https://raw.githubusercontent.com/irealva/tfjs-models/master/posenet/demos/camera.gif" alt="cameraDemo" style="width: 600px;"/>


### Demo 2: Coco Images

The [coco images](http://cocodataset.org/#home) demo shows how to estimate poses in images. It also illustrates the differences between the single-person and multi-person pose detection algorithms.

<img src="https://raw.githubusercontent.com/irealva/tfjs-models/master/posenet/demos/coco.gif" alt="cameraDemo" style="width: 600px;"/>


## Setup

cd into the demos folder:

```sh
cd posenet/demos
```

Install dependencies and prepare the build directory:

```sh
yarn
```

To watch files for changes, and launch a dev server:

```sh
yarn watch
```

## If you are developing posenet locally, and want to test the changes in the demos

Cd into the posenet folder:
```sh
cd posenet
```

Install dependencies:
```sh
yarn
```

Publish posenet locally:
```sh
yarn build && yalc publish
```

Cd into the demos and install dependencies:

```sh
cd demos
yarn
```

Link the local posenet to the demos:
```sh
yarn yalc link @tensorflow-models/posenet
```

Start the dev demo server:
```sh
yarn watch
```

To get future updates from the posenet source code:
```
# cd up into the posenet directory
cd ../
yarn build && yalc push
```
