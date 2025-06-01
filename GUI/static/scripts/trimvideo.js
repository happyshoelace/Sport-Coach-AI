import VideoEditor from "./VideoEditor-main/dist/VideoEditor.js";

export function trimVideo(inputFile) {
  console.log("Start trim");
  const editorContainer = document.getElementById("editor-container");
  //   editorContainer.innerHTML = ""; // clear any old UI
  editorContainer.style.width = "100%"; // ← your desired width
  editorContainer.style.height = "100%";
  const options = {
    src: inputFile,
    width: "100%", // force a 16∶9 viewport
    height: "100%",
    maxHeight: 450,
    limit: { maxDuration: 60 },
    onError: (error) => {
      console.error("VideoEditor error:", error);
    },
    onSave: (transformations, videoSrc) => {
      console.log("Transformations:", transformations);
      console.log("Video Source:", videoSrc);

      if (videoSrc) {
        const trimmedFile = new File([videoSrc], "trimmed_recording.webm", {
          type: "video/webm",
        });
        const dt = new DataTransfer();
        dt.items.add(trimmedFile);
        document.getElementById("video-file").files = dt.files;
        document
          .getElementById("video-file")
          .dispatchEvent(new Event("change"));

        // Store trim points if they exist
        if (transformations && transformations.time) {
          document.getElementById("video-trim-start").value =
            transformations.time.in || 0;
          document.getElementById("video-trim-end").value =
            transformations.time.out || 0;
        }

        // Clear the file input and enable upload
        document.getElementById("video-file").value = "";
        document.getElementById("upload-button").disabled = false;
      }

      // Clean up
      document.getElementById("editor-container").innerHTML = "";
    },
  };
  const editor = new VideoEditor(options);
  editor.render(editorContainer);
  editorContainer
    .querySelectorAll("button")
    .forEach((btn) => (btn.type = "button"));

  // ① Wait just a tick for all buttons to appear, then flip their type
  setTimeout(() => {
    editorContainer
      .querySelectorAll("button")
      .forEach((btn) => btn.setAttribute("type", "button"));
  }, 0);

  // ② Also intercept any accidental form-submit from inside here
  const form = document.getElementById("video-upload-form");
  form.addEventListener("submit", (e) => {
    if (e.submitter && e.submitter.id !== "upload-button") {
      e.preventDefault();
    }
  });
}
