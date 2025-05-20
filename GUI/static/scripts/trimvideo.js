import VideoEditor from './VideoEditor-main/dist/VideoEditor.js';

export function trimVideo(inputFile){
    const blobURL = URL.createObjectURL(inputFile);

    const editor = new VideoEditor({
        src: blobURL,
        onSave: async (transformations, videoSrc) => {
            URL.revokeObjectURL(blobURL);

            const {trimStart = 0, trimEnd = 0} = transformations;
            console.log('Transformations:', transformations);
            console.log('Video Source:', videoSrc);
            
            const dt = new DataTransfer();
            dt.items.add(videoSrc);
            document.getElementById('recorded-video').files = dt.files;
            document.getElementById('video-trim-start').value = trimStart;
            document.getElementById('video-trim-end').value = trimEnd;

            document.getElementById('video-file').value = '';
            document.getElementById('upload-button').disabled = false;
            document.getElementById('editor-container').innerHTML = '';
        }

});

    const editorContainer = document.getElementById('editor-container');
    editorContainer.innerHTML = '';
    editor.render(editorContainer);
}
