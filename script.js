import {
  FilesetResolver,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const MODEL_ASSET_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

const CONNECTIONS = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [24, 26],
  [26, 28],
  [27, 31],
  [28, 32],
];

const POSE_KEYS = {
  left: { shoulder: 11, hip: 23, ankle: 27, ear: 7 },
  right: { shoulder: 12, hip: 24, ankle: 28, ear: 8 },
};

const OFFSETS = {
  good: 18,
  warn: 40,
};

const ANGLE_WARN = 18;
const SIDE_VIEW_THRESHOLD = 0.18;

const startCameraButton = document.getElementById("startCameraButton");
const formStatus = document.getElementById("formStatus");
const primaryCue = document.getElementById("primaryCue");
const hipOffsetValue = document.getElementById("hipOffsetValue");
const bodyAngleValue = document.getElementById("bodyAngleValue");
const trackingPill = document.getElementById("trackingPill");
const hudMessage = document.getElementById("hudMessage");
const feedbackBadge = document.getElementById("feedbackBadge");
const feedbackText = document.getElementById("feedbackText");
const hipMetric = document.getElementById("hipMetric");
const confidenceMetric = document.getElementById("confidenceMetric");
const viewMetric = document.getElementById("viewMetric");

const video = document.getElementById("cameraFeed");
const poseCanvas = document.getElementById("poseCanvas");
const poseCtx = poseCanvas.getContext("2d");

let poseLandmarker;
let stream;
let lastVideoTime = -1;
let animationFrameId = 0;

function getPoint(landmarks, index) {
  const point = landmarks[index];
  return { x: point.x, y: point.y, visibility: point.visibility ?? 0 };
}

function selectBodySide(landmarks) {
  const left = POSE_KEYS.left;
  const right = POSE_KEYS.right;

  const leftScore =
    getPoint(landmarks, left.shoulder).visibility +
    getPoint(landmarks, left.hip).visibility +
    getPoint(landmarks, left.ankle).visibility;
  const rightScore =
    getPoint(landmarks, right.shoulder).visibility +
    getPoint(landmarks, right.hip).visibility +
    getPoint(landmarks, right.ankle).visibility;

  return leftScore >= rightScore ? "left" : "right";
}

function pointLineSignedDistance(point, start, end) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.hypot(dx, dy) || 1;
  return ((point.x - start.x) * dy - (point.y - start.y) * dx) / length;
}

function getBodyAngle(start, end) {
  const angle = Math.atan2(end.y - start.y, end.x - start.x) * (180 / Math.PI);
  return Math.abs(angle);
}

function toCanvasPoint(point) {
  return {
    x: (1 - point.x) * poseCanvas.width,
    y: point.y * poseCanvas.height,
  };
}

function drawLine(a, b, color, width = 4, dash = []) {
  poseCtx.beginPath();
  poseCtx.setLineDash(dash);
  poseCtx.lineWidth = width;
  poseCtx.strokeStyle = color;
  poseCtx.moveTo(a.x, a.y);
  poseCtx.lineTo(b.x, b.y);
  poseCtx.stroke();
  poseCtx.setLineDash([]);
}

function drawDot(point, color, radius = 7) {
  poseCtx.beginPath();
  poseCtx.fillStyle = color;
  poseCtx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  poseCtx.fill();
}

function setFeedback(state) {
  feedbackBadge.className = `feedback-badge ${state.badgeClass}`;
  feedbackBadge.textContent = state.badge;
  feedbackText.textContent = state.text;
  formStatus.textContent = state.status;
  primaryCue.textContent = state.cue;
  hudMessage.textContent = state.hud;
}

function setIdleState(message = "Waiting to start") {
  setFeedback({
    badgeClass: "feedback-neutral",
    badge: "Ready to analyse",
    text: "Start the camera and move into a side plank position.",
    status: message,
    cue: "Start the camera",
    hud: "Keep your full body side-on.",
  });
  trackingPill.textContent = "Tracker idle";
  hipOffsetValue.textContent = "-- px";
  bodyAngleValue.textContent = "-- deg";
  hipMetric.textContent = "--";
  confidenceMetric.textContent = "--";
  viewMetric.textContent = "Side profile needed";
}

function setTrackingState(message) {
  trackingPill.textContent = message;
}

function evaluatePosture(landmarks) {
  const side = selectBodySide(landmarks);
  const points = POSE_KEYS[side];
  const shoulder = getPoint(landmarks, points.shoulder);
  const hip = getPoint(landmarks, points.hip);
  const ankle = getPoint(landmarks, points.ankle);
  const ear = getPoint(landmarks, points.ear);

  const confidence = Math.min(
    shoulder.visibility,
    hip.visibility,
    ankle.visibility,
  );

  const sideProfileScore = Math.abs(
    getPoint(landmarks, 11).x - getPoint(landmarks, 12).x,
  );

  if (confidence < 0.45) {
    return {
      state: {
        badgeClass: "feedback-neutral",
        badge: "Move into frame",
        text: "Make sure shoulder, hip, and ankle stay visible.",
        status: "Low confidence",
        cue: "Step back slightly",
        hud: "Full side profile works best.",
      },
      confidence,
      sideProfileScore,
      hipOffset: null,
      bodyAngle: null,
      side,
      points: null,
      visible: false,
    };
  }

  if (sideProfileScore > SIDE_VIEW_THRESHOLD) {
    return {
      state: {
        badgeClass: "feedback-neutral",
        badge: "Turn sideways",
        text: "A clearer side view gives better plank analysis.",
        status: "Front-facing detected",
        cue: "Rotate your body 90 deg",
        hud: "Show your side profile to the camera.",
      },
      confidence,
      sideProfileScore,
      hipOffset: null,
      bodyAngle: null,
      side,
      points: null,
      visible: false,
    };
  }

  const shoulderCanvas = toCanvasPoint(shoulder);
  const hipCanvas = toCanvasPoint(hip);
  const ankleCanvas = toCanvasPoint(ankle);
  const earCanvas = toCanvasPoint(ear);

  const hipOffset = pointLineSignedDistance(hipCanvas, shoulderCanvas, ankleCanvas);
  const bodyAngle = getBodyAngle(shoulderCanvas, ankleCanvas);
  const headDrop = earCanvas.y - shoulderCanvas.y;

  let state;

  if (hipOffset > OFFSETS.warn) {
    state = {
      badgeClass: "feedback-bad",
      badge: "Raise hips",
      text: "Your hips are sagging below the ideal line.",
      status: "Needs correction",
      cue: "Lift through the core",
      hud: "Drive hips slightly upward.",
    };
  } else if (hipOffset > OFFSETS.good) {
    state = {
      badgeClass: "feedback-warn",
      badge: "Slightly low",
      text: "Lift your hips a little to create a straighter plank line.",
      status: "Almost there",
      cue: "Squeeze glutes",
      hud: "A little higher through the hips.",
    };
  } else if (hipOffset < -OFFSETS.warn) {
    state = {
      badgeClass: "feedback-bad",
      badge: "Lower hips",
      text: "Your hips are too high. Bring them down into one long line.",
      status: "Needs correction",
      cue: "Flatten the body line",
      hud: "Lower hips slightly.",
    };
  } else if (hipOffset < -OFFSETS.good) {
    state = {
      badgeClass: "feedback-warn",
      badge: "Slightly high",
      text: "Drop your hips a little to align shoulders, hips, and ankles.",
      status: "Almost there",
      cue: "Lower with control",
      hud: "Ease hips down a touch.",
    };
  } else if (Math.abs(bodyAngle) > ANGLE_WARN) {
    state = {
      badgeClass: "feedback-warn",
      badge: "Reset line",
      text: "Aim for a flatter shoulder-to-ankle angle.",
      status: "Adjust posture",
      cue: "Lengthen through the body",
      hud: "Think head to heel line.",
    };
  } else if (headDrop > 36) {
    state = {
      badgeClass: "feedback-warn",
      badge: "Neck neutral",
      text: "Bring your head in line with the rest of your body.",
      status: "Small adjustment",
      cue: "Look slightly ahead",
      hud: "Keep neck long and neutral.",
    };
  } else {
    state = {
      badgeClass: "feedback-good",
      badge: "Strong line",
      text: "Nice plank position. Keep shoulders, hips, and ankles stacked.",
      status: "Good form",
      cue: "Hold steady",
      hud: "Maintain that straight line.",
    };
  }

  return {
    state,
    confidence,
    sideProfileScore,
    hipOffset,
    bodyAngle,
    side,
    points: { shoulder, hip, ankle, ear },
    canvasPoints: { shoulder: shoulderCanvas, hip: hipCanvas, ankle: ankleCanvas },
    visible: true,
  };
}

function drawSkeleton(landmarks) {
  poseCtx.strokeStyle = "rgba(255,255,255,0.20)";
  poseCtx.lineWidth = 2;

  for (const [startIndex, endIndex] of CONNECTIONS) {
    const start = landmarks[startIndex];
    const end = landmarks[endIndex];
    if (!start || !end) {
      continue;
    }

    const a = toCanvasPoint(start);
    const b = toCanvasPoint(end);
    drawLine(a, b, "rgba(255,255,255,0.16)", 2);
  }
}

function drawAnalysis(landmarks, analysis) {
  poseCtx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
  drawSkeleton(landmarks);

  if (!analysis.visible || !analysis.points) {
    return;
  }

  const { shoulder, hip, ankle } = analysis.canvasPoints;
  const colors = {
    good: "#53d48e",
    warn: "#ffb347",
    bad: "#ff6b6b",
    neutral: "#f6b93b",
  };

  let lineColor = colors.good;
  if (analysis.state.badgeClass === "feedback-warn") {
    lineColor = colors.warn;
  } else if (analysis.state.badgeClass === "feedback-bad") {
    lineColor = colors.bad;
  }

  drawLine(shoulder, ankle, "rgba(255,255,255,0.28)", 3, [10, 8]);
  drawLine(shoulder, hip, lineColor, 5);
  drawLine(hip, ankle, lineColor, 5);
  drawLine(hip, {
    x: hip.x,
    y: shoulder.y + ((ankle.y - shoulder.y) * ((hip.x - shoulder.x) / ((ankle.x - shoulder.x) || 1))),
  }, lineColor, 2, [6, 6]);

  drawDot(shoulder, "#f6b93b", 8);
  drawDot(hip, lineColor, 9);
  drawDot(ankle, "#f6b93b", 8);
}

function updateDashboard(analysis) {
  setFeedback(analysis.state);

  const confidencePercent = `${Math.round(analysis.confidence * 100)}%`;
  confidenceMetric.textContent = confidencePercent;
  viewMetric.textContent =
    analysis.sideProfileScore <= SIDE_VIEW_THRESHOLD ? "Side profile ok" : "Rotate more";

  if (analysis.visible && analysis.hipOffset !== null && analysis.bodyAngle !== null) {
    const hipOffsetRounded = Math.round(Math.abs(analysis.hipOffset));
    hipOffsetValue.textContent = `${hipOffsetRounded} px`;
    bodyAngleValue.textContent = `${Math.round(analysis.bodyAngle)} deg`;
    hipMetric.textContent =
      analysis.hipOffset > 0
        ? `${hipOffsetRounded}px low`
        : analysis.hipOffset < 0
          ? `${hipOffsetRounded}px high`
          : "On line";
    setTrackingState("Tracking live");
  } else {
    hipOffsetValue.textContent = "-- px";
    bodyAngleValue.textContent = "-- deg";
    hipMetric.textContent = "--";
    setTrackingState("Adjusting view");
  }
}

async function setupPoseLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_ASSET_URL },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.55,
    minPosePresenceConfidence: 0.55,
    minTrackingConfidence: 0.55,
  });
}

function syncCanvasSize() {
  if (!video.videoWidth || !video.videoHeight) {
    return;
  }

  poseCanvas.width = video.videoWidth;
  poseCanvas.height = video.videoHeight;
}

function clearCanvas() {
  poseCtx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
}

function detectPose() {
  if (!poseLandmarker || video.readyState < 2) {
    animationFrameId = requestAnimationFrame(detectPose);
    return;
  }

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = poseLandmarker.detectForVideo(video, performance.now());
    const landmarks = result.landmarks?.[0];

    if (landmarks) {
      const analysis = evaluatePosture(landmarks);
      drawAnalysis(landmarks, analysis);
      updateDashboard(analysis);
    } else {
      clearCanvas();
      setFeedback({
        badgeClass: "feedback-neutral",
        badge: "Find your pose",
        text: "Step back until the camera sees your body clearly.",
        status: "Pose not found",
        cue: "Show full body line",
        hud: "Shoulders to ankles in frame.",
      });
      trackingPill.textContent = "Searching";
      hipOffsetValue.textContent = "-- px";
      bodyAngleValue.textContent = "-- deg";
      hipMetric.textContent = "--";
      confidenceMetric.textContent = "--";
      viewMetric.textContent = "Need clearer frame";
    }
  }

  animationFrameId = requestAnimationFrame(detectPose);
}

async function enableCamera() {
  if (stream) {
    return;
  }

  startCameraButton.disabled = true;
  startCameraButton.textContent = "Starting...";
  trackingPill.textContent = "Loading tracker";
  primaryCue.textContent = "Preparing camera";

  try {
    await setupPoseLandmarker();
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();
    syncCanvasSize();
    setFeedback({
      badgeClass: "feedback-neutral",
      badge: "Camera ready",
      text: "Move into a side plank so we can analyse your body line.",
      status: "Camera live",
      cue: "Set your plank",
      hud: "Turn sideways and hold still for a moment.",
    });
    confidenceMetric.textContent = "--";
    viewMetric.textContent = "Checking angle";
    startCameraButton.textContent = "Camera Running";
    detectPose();
  } catch (error) {
    console.error(error);
    startCameraButton.disabled = false;
    startCameraButton.textContent = "Retry Camera Analysis";
    setFeedback({
      badgeClass: "feedback-bad",
      badge: "Camera blocked",
      text: "Allow camera access, then reload or retry.",
      status: "Camera unavailable",
      cue: "Grant permission",
      hud: "Use localhost or https if needed.",
    });
    trackingPill.textContent = "Camera unavailable";
    confidenceMetric.textContent = "--";
    viewMetric.textContent = "Permission needed";
  }
}

startCameraButton.addEventListener("click", enableCamera);
window.addEventListener("resize", syncCanvasSize);
window.addEventListener("beforeunload", () => {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
  }

  if (stream) {
    for (const track of stream.getTracks()) {
      track.stop();
    }
  }
});

setIdleState();
