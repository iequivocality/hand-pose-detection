// @ts-nocheck
import { ScatterGL } from "scatter-gl";
import { DEFAULT_LINE_WIDTH, STATE, VIDEO_SIZE } from "../utils/params";
import { isMobile } from "../utils";

// position in the input.
const ANCHOR_POINTS = [[0, 0, 0], [0, 0.1, 0], [-0.1, 0, 0], [-0.1, -0.1, 0]];

const fingerLookupIndices = {
  thumb: [0, 1, 2, 3, 4],
  indexFinger: [0, 5, 6, 7, 8],
  middleFinger: [0, 9, 10, 11, 12],
  ringFinger: [0, 13, 14, 15, 16],
  pinky: [0, 17, 18, 19, 20],
}; // for rendering each finger as a polyline

const connections = [
  [0, 1], [1, 2], [2, 3], [3,4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13,14], [14, 15], [15, 16],
  [0, 17], [17, 18],[18, 19], [19,20]
];

type ScatterGLContext = {
  scatterGLEl: HTMLElement;
  scatterGL: ScatterGL;
  scatterGLHasInitialized: boolean;
}

function createScatterGLContext(selectors: string) {
  const scatterGLEl = document.querySelector(selectors) as HTMLElement;
  console.log("createScatterGLContext", scatterGLEl)
  return {
    scatterGLEl,
    scatterGL: new ScatterGL(scatterGLEl, {
      'rotateOnStart': true,
      'selectEnabled': false,
      'styles': {polyline: {defaultOpacity: 1, deselectedOpacity: 1}}
    }),
    scatterGLHasInitialized: false,
  };
}

type CameraParam = {
  targetFPS: number,
  sizeOption: keyof typeof VIDEO_SIZE,
}

export class Camera {
  video: HTMLVideoElement;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  scatterGLCtxtLeftHand: ScatterGLContext;
  scatterGLCtxtRightHand: ScatterGLContext;

  constructor() {
    this.video = document.getElementById('video') as HTMLVideoElement;
    this.canvas = document.getElementById('output') as HTMLCanvasElement;
    this.ctx = this.canvas.getContext('2d') as CanvasRenderingContext2D;
    this.scatterGLCtxtLeftHand = createScatterGLContext('#scatter-gl-container-left');
    this.scatterGLCtxtRightHand = createScatterGLContext('#scatter-gl-container-right');
  }

  static async setupCamera(cameraParam: CameraParam) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
          'Browser API navigator.mediaDevices.getUserMedia not available');
    }

    const {targetFPS, sizeOption} = cameraParam;
    const $size = VIDEO_SIZE[sizeOption];
    const videoConfig = {
      'audio': false,
      'video': {
        facingMode: 'user',
        // Only setting the video to a specified size for large screen, on
        // mobile devices accept the default size.
        width: isMobile() ? VIDEO_SIZE['360 X 270'].width : $size.width,
        height: isMobile() ? VIDEO_SIZE['360 X 270'].height :
                             $size.height,
        frameRate: {
          ideal: targetFPS,
        }
      }
    };

    const stream = await navigator.mediaDevices.getUserMedia(videoConfig);

    const camera = new Camera();
    camera.video.srcObject = stream;

    await new Promise((resolve) => {
      camera.video.onloadedmetadata = () => {
        resolve();
      };
    });

    camera.video.play();

    const videoWidth = camera.video.videoWidth;
    const videoHeight = camera.video.videoHeight;
    // Must set below two lines, otherwise video element doesn't show.
    camera.video.width = videoWidth;
    camera.video.height = videoHeight;

    camera.canvas.width = videoWidth;
    camera.canvas.height = videoHeight;
    const canvasContainer = document.querySelector('.canvas-wrapper') as HTMLCanvasElement;
    canvasContainer.style = `width: ${videoWidth}px; height: ${videoHeight}px`;

    // Because the image from camera is mirrored, need to flip horizontally.
    camera.ctx.translate(camera.video.videoWidth, 0);
    camera.ctx.scale(-1, 1);

    console.log([this.scatterGLCtxtLeftHand, this.scatterGLCtxtRightHand] )
    for (const ctxt of [camera.scatterGLCtxtLeftHand, camera.scatterGLCtxtRightHand]) {
      ctxt.scatterGLEl.style =
          `width: ${videoWidth / 2}px; height: ${videoHeight / 2}px;`;
      ctxt.scatterGL.resize();

      ctxt.scatterGLEl.style.display =
          STATE.modelConfig.render3D ? 'inline-block' : 'none';
    }

    return camera;
  }

  drawCtx() {
    this.ctx.drawImage(
        this.video, 0, 0, this.video.videoWidth, this.video.videoHeight);
  }

  clearCtx() {
    this.ctx.clearRect(0, 0, this.video.videoWidth, this.video.videoHeight);
  }

  /**
   * Draw the keypoints on the video.
   * @param hands A list of hands to render.
   */
  drawResults(hands) {
    // Sort by right to left hands.
    hands.sort((hand1, hand2) => {
      if (hand1.handedness < hand2.handedness) return 1;
      if (hand1.handedness > hand2.handedness) return -1;
      return 0;
    });

    // Pad hands to clear empty scatter GL plots.
    while (hands.length < 2) hands.push({});

    for (let i = 0; i < hands.length; ++i) {
      // Third hand and onwards scatterGL context is set to null since we
      // don't render them.
      const ctxt = [this.scatterGLCtxtLeftHand, this.scatterGLCtxtRightHand][i];
      this.drawResult(hands[i], ctxt);
    }
  }

  /**
   * Draw the keypoints on the video.
   * @param hand A hand with keypoints to render.
   * @param ctxt Scatter GL context to render 3D keypoints to.
   */
  drawResult(hand, ctxt) {
    if (hand.keypoints != null) {
      this.drawKeypoints(hand.keypoints, hand.handedness);
    }
    // Don't render 3D hands after first two.
    if (ctxt == null) {
      return;
    }
    if (hand.keypoints3D != null && STATE.modelConfig.render3D) {
      this.drawKeypoints3D(hand.keypoints3D, hand.handedness, ctxt);
    } else {
      // Clear scatter plot.
      this.drawKeypoints3D([], '', ctxt);
    }
  }

  /**
   * Draw the keypoints on the video.
   * @param keypoints A list of keypoints.
   * @param handedness Label of hand (either Left or Right).
   */
   drawKeypoints(keypoints, handedness) {
    const keypointsArray = keypoints;
    this.ctx.fillStyle = handedness === 'Left' ? 'Red' : 'Blue';
    this.ctx.strokeStyle = 'White';
    this.ctx.lineWidth = DEFAULT_LINE_WIDTH;

    for (let i = 0; i < keypointsArray.length; i++) {
      const y = keypointsArray[i].x;
      const x = keypointsArray[i].y;
      this.drawPoint(x - 2, y - 2, 3);
    }

    const fingers = Object.keys(fingerLookupIndices);
    for (let i = 0; i < fingers.length; i++) {
      const finger = fingers[i];
      const points = fingerLookupIndices[finger].map(idx => keypoints[idx]);
      this.drawPath(points, false);
    }
  }

  drawPath(points, closePath) {
    const region = new Path2D();
    region.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      const point = points[i];
      region.lineTo(point.x, point.y);
    }

    if (closePath) {
      region.closePath();
    }
    this.ctx.stroke(region);
  }

  drawPoint(y, x, r) {
    this.ctx.beginPath();
    this.ctx.arc(x, y, r, 0, 2 * Math.PI);
    this.ctx.fill();
  }

  drawKeypoints3D(keypoints, handedness, ctxt) {
    const scoreThreshold = STATE.modelConfig.scoreThreshold || 0;
    const pointsData =
        keypoints.map(keypoint => ([-keypoint.x, -keypoint.y, -keypoint.z]));

    const dataset =
        new ScatterGL.Dataset([...pointsData, ...ANCHOR_POINTS]);

    ctxt.scatterGL.setPointColorer((i) => {
      if (keypoints[i] == null || keypoints[i].score < scoreThreshold) {
        // hide anchor points and low-confident points.
        return '#ffffff';
      }
      return handedness === 'Left' ? '#ff0000' : '#0000ff';
    });

    if (!ctxt.scatterGLHasInitialized) {
      ctxt.scatterGL.render(dataset);
    } else {
      ctxt.scatterGL.updateDataset(dataset);
    }
    const sequences = connections.map(pair => ({indices: pair}));
    ctxt.scatterGL.setSequences(sequences);
    ctxt.scatterGLHasInitialized = true;
  }
}