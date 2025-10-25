const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startRecordingButton = document.getElementById('startRecording');
const stopRecordingButton = document.getElementById('stopRecording');

let mediaRecorder;
let recordedChunks = [];

const faceDetector = new FaceDetector({ locateFaces: true });

async function startVideo() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (error) {
        console.error('Error accessing webcam:', error);
    }
}

async function detectFaces() {
    const predictions = await faceDetector.send({ image: video });
    return predictions;
}

function drawFaces(predictions) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    predictions.forEach(prediction => {
        const { x, y, width, height } = prediction.boundingBox;
        ctx.filter = 'blur(12px)';
        ctx.drawImage(video, x, y, width, height, x, y, width, height);
    });
}

function processVideo() {
    detectFaces().then(predictions => {
        drawFaces(predictions);
    });
}

video.addEventListener('play', () => {
    requestAnimationFrame(processVideo);
});

startVideo();

function startRecording() {
    const stream = canvas.captureStream(); // Capture the canvas stream
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    mediaRecorder.ondataavailable = event => {
        if (mediaRecorder.state === 'recording') {
            recordedChunks.push(event.data);
        }
    };
    mediaRecorder.start();
}

function stopRecording() {
    mediaRecorder.stop();
    mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        console.log('Blob size:', blob.size); // Check the size of the blob
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'blurred-video.webm';
        a.click();
        URL.revokeObjectURL(url);
        recordedChunks = [];
    };
}

startRecordingButton.addEventListener('click', startRecording);
stopRecordingButton.addEventListener('click', stopRecording);