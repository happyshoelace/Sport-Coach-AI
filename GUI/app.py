from flask import Flask, render_template, request, redirect, jsonify, send_from_directory
from wrapper import video_name_to_predictions
import threading
import uuid
from flask import current_app
import os
import json

app = Flask(__name__)

# Configure proper MIME type for JavaScript modules
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['MIME_TYPES'] = {
    '.js': 'application/javascript',
    '.mjs': 'application/javascript'
}

jobs = {}
dominant_hand = "right"

def process_video_background(job_id, video_server_path, filename_only, hand_for_job, app_context):
    with app_context:
        try:
            print(f"Background processing started for job_id: {job_id}, file: {filename_only}, hand: {hand_for_job}")
            # Pass only the filename to your model, not the URL
            probability, index, total_frame_predictions = video_name_to_predictions(filename_only, hand_for_job)
            classes = ["En Garde", "Fleche", "Lunge", "Step"]
            results_data = {
                'video_url': f"/uploads/{filename_only}_annotated.webm",
                'footworkClass': classes[index],
                'classConfidence': float(probability),
                'all_predictions_json': total_frame_predictions.tolist()
            }
            jobs[job_id] = {'status': 'complete', 'data': results_data}
            # print(f"Job {job_id} completed. Results: {results_data}")
        except Exception as e:
            print(f"Error during background processing for job {job_id}: {e}")
            import traceback
            traceback.print_exc()
            jobs[job_id] = {'status': 'error', 'message': str(e)}

@app.route('/static/scripts/<path:filename>')
def serve_static(filename):
    response = send_from_directory('static/scripts', filename)
    if filename.endswith('.js'):
        response.headers['Content-Type'] = 'application/javascript'
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index')
def homePage():
    return render_template('index.html')

@app.route('/blog')
def blogPage():
    return render_template('blog.html')

@app.route('/login')
def loginPage():
    return render_template('login.html')

@app.route('/contact')
def contactPage():
    return render_template('contact.html')

@app.route('/modelinputhand')
def modelInputHandPage():
    return render_template('modelinputhand.html')

@app.route('/modelvideoInput', methods=['POST'])
def modelVideoInputPage():
    if request.method == 'POST':
        global dominant_hand; dominant_hand = request.form.get('hand')
        print(f"Dominant hand selected: {dominant_hand}")
    return render_template('modelvideoInput.html')


@app.route('/trimvideo', methods=['POST'])
def trimVideoPage():
    if request.method == 'POST':
        video = request.files.get('file')
        
    return render_template('trimvideo.html', video=video)

# @app.route('/modeloutput', methods=['POST'])
# def modelOutputPage():
#     if request.method == 'POST':
#         global dominant_hand
#         print(request.files.get('file'))
#         video = request.files.get('file')
#         # We can pop a post request to AWS here and that will fill the gaps for the template page 
#     return render_template('modeloutput.html', jointsVideo=video, footworkClass="En Garde", classConfidence=100) 
from werkzeug.utils import secure_filename
from flask import url_for

# at top
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/feedback', methods=['POST'])
def modelOutputPage():
    if 'file' not in request.files:
        # handle missing file…
        return redirect(url_for('modelVideoInputPage'))
    file = request.files['file']
    if file.filename == '':
        # handle empty filename…
        return redirect(url_for('modelVideoInputPage'))

    # save the trimmed video
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    video_url = url_for('uploaded_file', filename=filename)
    

    # don't have to provide hand
    probability, index, total_frame_predictions = video_name_to_predictions(video_url, dominant_hand)
    classes = ["En Garde", "Fleche", "Lunge", "Step"]
    print(video_url)

    # now pass video_url (a string) to your template
    return render_template(
        'modeloutput.html',
        video_url=url_for('uploaded_file', filename='output.mp4'),
        footworkClass=classes[index],
        classConfidence=probability,
        all_predictions_json_str=json.dumps(total_frame_predictions.tolist())
    )

@app.route('/initiate_processing', methods=['POST'])
def initiate_processing_route():
    global dominant_hand
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    if file:
        filename = secure_filename(file.filename)
        video_server_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_server_path)
        print(f"File saved to: {video_server_path}")

        job_id = str(uuid.uuid4())
        jobs[job_id] = {'status': 'processing', 'data': None}

        app_context = current_app.app_context()
        thread = threading.Thread(target=process_video_background, args=(job_id, video_server_path, filename, dominant_hand, app_context))
        thread.start()

        print(f"Processing initiated for job_id: {job_id} with hand: {dominant_hand}")
        return jsonify({'job_id': job_id})
    return jsonify({'error': 'File upload failed'}), 500

@app.route('/processing_status/<job_id>', methods=['GET'])
def processing_status_route(job_id):
    job_info = jobs.get(job_id)
    if not job_info:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job_info)

@app.route('/modeloutput_display')
def model_output_display_page():
    video_url = request.args.get('video_url')
    footwork_class = request.args.get('footworkClass')
    class_confidence = request.args.get('classConfidence')
    all_predictions_json_str = request.args.get('all_predictions_json')
    # Convert class_confidence to float if possible
    try:
        class_confidence = float(class_confidence)*100
    except (TypeError, ValueError):
        class_confidence = 0.0
    return render_template('modeloutput.html',
                           video_url=video_url,
                           footworkClass=footwork_class,
                           classConfidence=class_confidence,
                           all_predictions_json_str=all_predictions_json_str)

@app.route('/loading')
def loading_page_route():
    return render_template('loading.html')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)