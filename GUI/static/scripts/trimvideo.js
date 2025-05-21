import VideoEditor from './VideoEditor-main/dist/VideoEditor.js';

export function trimVideo(inputFile) {
    const options = {
        src: inputFile,  // VideoEditor accepts Blob/File directly according to docs
        maxHeight: 450,
        limit: { maxDuration: 60 },  // 60 second max duration
        onError: (error) => {
            console.error('VideoEditor error:', error);
        },
        onSave: (transformations, videoSrc) => {
            console.log('Transformations:', transformations);
            console.log('Video Source:', videoSrc);
            
            if (videoSrc) {
                const dt = new DataTransfer();
                const trimmedFile = new File([videoSrc], "trimmed_recording.webm", { type: "video/webm" });
                dt.items.add(trimmedFile);
                document.getElementById('recorded-video').files = dt.files;
                
                // Store trim points if they exist
                if (transformations && transformations.time) {
                    document.getElementById('video-trim-start').value = transformations.time.in || 0;
                    document.getElementById('video-trim-end').value = transformations.time.out || 0;
                }

                // Clear the file input and enable upload
                document.getElementById('video-file').value = '';
                document.getElementById('upload-button').disabled = false;
            }

            // Clean up
            document.getElementById('editor-container').innerHTML = '';
        }
    };

        const editor = new VideoEditor(options);
        const editorContainer = document.getElementById('editor-container');
        editorContainer.innerHTML = '';
        editor.render(editorContainer);
    };
