from flask import Flask, render_template, request, redirect, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index')
def homePage():
    return render_template('index.html')

@app.route('/blog')
def blogPage():
    return render_template('blog.html')

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

@app.route('/modeloutput', methods=['POST'])
def modelOutputPage():
    if request.method == 'POST':
        global dominant_hand
        print(request.files.get('file'))
        video = request.files.get('file')
        # We can pop a post request to AWS here and that will fill the gaps for the template page 
    return render_template('modeloutput.html', jointsVideo=video, footworkClass="En Garde", classConfidence=100) 