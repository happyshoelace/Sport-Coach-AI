import { trimVideo } from "./trimvideo.js";
let mediaRecorder;
let recordedChunks = [];

const preview = document.getElementById("preview");
const record_button = document.getElementById("record-button");
const stop_button = document.getElementById("stop-button");
const downloadLink = document.getElementById("download-link");
const uploadForm = document.getElementById("video-upload-form");
const uploadButton = document.getElementById("upload-button");
const videoFileInput = document.getElementById("video-file");
const recordedFile = document.getElementById("recorded-video");

navigator.mediaDevices.getUserMedia({video:true, audio:false})
    .then(stream => {
        preview.srcObject = stream;

        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = function(e) {
            if (e.data.size >0) {
                recordedChunks.push(e.data);
            }
        };

        mediaRecorder.onstop = () => {
            const blob = new Blob(recordedChunks, {type: "video/webm"});
            const file = new File([blob], "recording.webm", {type: "video/webm"});
            trimVideo(file);
            // const dt = new DataTransfer();
            // dt.items.add(file);
            // recordedFile.files = dt.files;
    
            // document.getElementById('video-file').value = '';
            // uploadButton.disabled = false;


    
        };
    })
    .catch(error => {
        console.error("Error accessing media devices.", error);
    });

    record_button.onclick = () => {
        recordedChunks = [];
        mediaRecorder.start();
        record_button.disabled = true;
        stop_button.disabled = false;
    };

    stop_button.onclick = () => {
        mediaRecorder.stop();
        record_button.disabled = false;
        stop_button.disabled = true;
    }

function toggleUploadButton() {
    const hasVideoFile = videoFileInput.files.length > 0;
    const hasRecordedFile = recordedFile.files.length > 0;
    uploadButton.disabled = !(hasVideoFile || hasRecordedFile);
}

videoFileInput.addEventListener('change',  toggleUploadButton);
recordedFile.addEventListener('change', toggleUploadButton);
