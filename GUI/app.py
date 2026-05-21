from flask import Flask, render_template, request, redirect, jsonify, send_from_directory
from wrapper import video_name_to_predictions, save_video_with_keypoints
import os

app = Flask(__name__)

# Configure proper MIME type for JavaScript modules
app.config['MIME_TYPES'] = {
    '.js': 'application/javascript',
    '.mjs': 'application/javascript'
}

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

@app.route('/feedback', methods=['GET','POST'])
def modelOutputPage():
    if request.method == 'GET':
        print("GET request")
        return redirect(url_for('modelInputHandPage'))
    print(request.files)
    if 'fileV' in request.files and request.files['fileV'].filename != '':
        file = request.files['fileV']
    elif 'fileR' in request.files and request.files['fileR'].filename != '':
        file = request.files['fileR']
    else:
        print("No file uploaded")
        # handle missing file…
        return redirect(url_for('modelVideoInputPage'))
    if file.filename == '':
        print("Empty filename")
        # handle empty filename…
        return redirect(url_for('modelVideoInputPage'))

    # save the trimmed video
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    video_url = url_for('uploaded_file', filename=filename)

    # don't have to provide hand
    probability, index, total_frame_predictions = video_name_to_predictions(video_url, dominant_hand)
    save_video_with_keypoints(video_url, video_url)

    for x, i in enumerate(total_frame_predictions):
        print(x, i)
        print(total_frame_predictions[x])
    classes = ["En Garde", "Fleche", "Lunge", "Step"]

    # now pass video_url (a string) to your template
    return render_template('modeloutput.html',
                           video_url=video_url,
                           footworkClass=classes[index],
                           classConfidence=probability*100,
                           all_predictions=total_frame_predictions,
                           enGardeConfidence=total_frame_predictions[0][0],
                           flecheConfidence=total_frame_predictions[0][1],
                           lungeConfidence=total_frame_predictions[0][2],
                           stepConfidence=total_frame_predictions[0][3])

if __name__ == "__main__":
    app.run(debug=True)