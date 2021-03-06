// https://docs.opencv.org/3.4.0/df/d6c/tutorial_js_face_detection_camera.html

// Video Settings
const width  = 640;
const height = 480;
const fps = 30;

// Globals
let videoCapture = null;
let stream = null;
let isStreaming = false;
let matSrc  = null;
let matDst  = null;
let matGrey = null;
let faces = null;
let classifier = null;

// Elements
const videoElem  = document.getElementById('video');
const canvasElem = document.getElementById('canvas');
const startButtonElem = document.getElementById('start');
const stopButtonElem  = document.getElementById('stop');
const statusElem = document.getElementById('status');

// https://docs.opencv.org/3.4.0/haarcascade_frontalface_default.xml
const faceCascadeFile = './haarcascade_frontalface_default.xml';

// ================================================================================

/** On OpenCV.js Loaded */
function onCvLoaded() {
  console.log('on OpenCV.js Loaded', cv);
  
  cv.onRuntimeInitialized = onReady;  // Not Working...
  statusElem.innerText = 'On OpenCV.js Loaded';
}

/** On Ready */
function onReady() {
  console.log('On Ready');
  
  // Set Element Size
  videoElem.width  = canvasElem.width  = width;
  videoElem.height = canvasElem.height = height;
  // Start Video Capture
  videoCapture = new cv.VideoCapture(videoElem);
  // Load XML File With XHR
  const utils = new Utils('error-message');  // Set Element ID
  utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
    console.log('Face Cascade File Loaded');
  });
  // Set Button Events
  startButtonElem.addEventListener('click', onStart);
  startButtonElem.disabled = false;
  stopButtonElem.addEventListener('click', onStop);
  stopButtonElem.disabled  = true;
  
  statusElem.innerText = 'Ready';
};

/** On Window Loaded */
window.addEventListener('load', () => {
  console.log('Window Loaded');
  
  onReady();  // cv.onRuntimeInitialized Is Not Working, So This Is Fallback
});

/** On Start */
function onStart() {
  navigator.mediaDevices.getUserMedia({
    video: true,
    audio: false
  })
    .then((_stream) => {
      console.log('On Start : Success');
      
      stream = videoElem.srcObject = _stream;
      videoElem.play();
      
      matSrc  = new cv.Mat(height, width, cv.CV_8UC4);  // For Video Capture
      matDst  = new cv.Mat(height, width, cv.CV_8UC4);  // For Canvas Preview
      matGrey = new cv.Mat();
      faces = new cv.RectVector();
      classifier = new cv.CascadeClassifier();
      // Load Pre-Trained Classifiers
      classifier.load(faceCascadeFile);
      
      // Start Process Video
      setTimeout(processVideo, 0);
      
      isStreaming = true;
      startButtonElem.disabled = true;
      stopButtonElem.disabled  = false;
    })
    .catch((error) => {
      console.error('On Start : Error', error);
    });
}

/** On Stop */
function onStop() {
  console.log('On Stop');
  
  videoElem.pause();
  videoElem.srcObject = null;
  stream.getVideoTracks()[0].stop();
  
  isStreaming = false;
  startButtonElem.disabled = false;
  stopButtonElem.disabled  = true;
}

/** Process Video */
function processVideo() {
  if(!isStreaming) {
    console.log('Process Video : Streaming Stopped');
    
    matSrc.delete();
    matDst.delete();
    matGrey.delete();
    faces.delete();
    classifier.delete();
    return;
  }
  
  const begin = Date.now();
  videoCapture.read(matSrc);  // Capture Video Image To Mat Src
  matSrc.copyTo(matDst);  // Copy Src To Dst
  cv.cvtColor(matDst, matGrey, cv.COLOR_RGBA2GRAY, 0);  // Get Grey Image
  classifier.detectMultiScale(matGrey, faces, 1.1, 3, 0);  // Detect Faces
  // Draw Faces Rectangle
  for(let i = 0; i < faces.size(); ++i) {
    const face = faces.get(i);
    const point1 = new cv.Point(face.x, face.y);
    const point2 = new cv.Point(face.x + face.width, face.y + face.height);
    cv.rectangle(matDst, point1, point2, [255, 0, 0, 255]);
  }
  cv.imshow('canvas', matDst);  // Set Element ID
  
  // Loop
  const delay = 1000 / fps - (Date.now() - begin);
  setTimeout(processVideo, delay);
}