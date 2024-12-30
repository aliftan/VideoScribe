from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip
import whisper
import threading
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'mkv'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('temp', exist_ok=True)

# Store transcription status
transcription_status = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_audio(video_path, audio_path):
    """Extract audio from video file"""
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        return True
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return False


def transcribe_audio(audio_path, model_size="base"):
    """Transcribe audio file using Whisper"""
    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return None


def process_video(video_path, task_id):
    """Process video file and update status"""
    try:
        transcription_status[task_id]['status'] = 'extracting_audio'

        # Extract audio
        audio_path = os.path.join('temp', f'audio_{task_id}.mp3')
        if not extract_audio(video_path, audio_path):
            transcription_status[task_id]['status'] = 'failed'
            return

        transcription_status[task_id]['status'] = 'transcribing'

        # Transcribe audio
        transcript = transcribe_audio(audio_path)

        # Clean up
        os.remove(audio_path)
        os.remove(video_path)

        if transcript:
            transcription_status[task_id]['status'] = 'completed'
            transcription_status[task_id]['transcript'] = transcript
        else:
            transcription_status[task_id]['status'] = 'failed'

    except Exception as e:
        transcription_status[task_id]['status'] = 'failed'
        transcription_status[task_id]['error'] = str(e)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Generate unique task ID
        task_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save file
        filename = secure_filename(file.filename)
        video_path = os.path.join(
            app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        file.save(video_path)

        # Initialize status
        transcription_status[task_id] = {
            'status': 'starting',
            'filename': filename
        }

        # Start processing in background
        thread = threading.Thread(
            target=process_video, args=(video_path, task_id))
        thread.start()

        return jsonify({
            'task_id': task_id,
            'message': 'Video uploaded successfully'
        })

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/status/<task_id>')
def get_status(task_id):
    if task_id not in transcription_status:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(transcription_status[task_id])


if __name__ == '__main__':
    app.run(debug=True)
