let mediaRecorder;
let recordedChunks = [];

const preview = document.getElementById("preview");
const record_button = document.getElementById("record-button");
const stop_button = document.getElementById("stop-button");
const downloadLink = document.getElementById("download-link");

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
            const url = URL.createObjectURL(blob);
            downloadLink.href = url;
            downloadLink.download = "recording.webm";
            downloadLink.style.display = "inline";
            downloadLink.textContent = "Download recording";
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

// I'm sure at some point we will want to be able to save the video to be able to put it in the model but this will do for now