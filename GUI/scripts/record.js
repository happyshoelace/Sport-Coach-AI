let mediaRecorder;
let recordedChunks = [];

const preview = document.getElementById("preview");
const record_button = document.getElementById("record-button");
const stop_button = document.getElementById("stop-button");
const downloadLink = document.getElementById("downloadLink");

navigator.mediaDevices.getUserMedia({video:true, audio:false})
    .then(stream => {
        preview.srcObject = stream;

        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = e => {
            if (e.data.size >0) {
                recordedChunks.push(e.data);
            }
        };

        mediatRecorder.onstop = () => {
            const blob = new Blob(recordedChunks, {type: "video/webm"});
            const url = URL.createObjectURL(Blob);
            downloadLink.href = url;
            downloadLink.download = "recording.webm";
            downloadLink.style.display = "inline";
            downloadLink.textContent = "Download recording";
        };
    })
    .catch(error => {
        console.error("Error accessing media devices.", error);
    });

    start_button.onclick = () => {
        recordedChunks = [];
        mediaRecorder.start();
        start_button.disabled = true;
        stop_button.disabled = false;
    };

    stop_button.onclick = () => {
        mediaRecorder.stop();
        start_button.disabled = false;
        stop_button.disabled = true;
    }