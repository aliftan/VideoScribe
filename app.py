from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip
import whisper
import threading
from datetime import datetime
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv',
                      'mkv', 'mp3', 'wav', 'm4a', 'aac', 'ogg'}
VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'mkv'}
AUDIO_EXTENSIONS = {'mp3', 'wav', 'm4a', 'aac', 'ogg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('temp', exist_ok=True)

# Store transcription status
transcription_status = {}


def is_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS


def is_audio(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in AUDIO_EXTENSIONS


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


def transcribe_audio(audio_path, model_size="base", translate=False):
    """Transcribe audio file using Whisper with language detection"""
    try:
        model = whisper.load_model(model_size)

        # First detect the language
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        detected_language = max(probs, key=probs.get)

        # Now transcribe with all the options
        options = {
            "task": "translate" if translate else "transcribe",
            "language": detected_language,
            "verbose": True
        }

        result = model.transcribe(audio_path, **options)

        return {
            "text": result["text"],
            "language": detected_language,
            "language_probability": probs[detected_language]
        }
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return None


def process_file(file_path, task_id):
    """Process video or audio file and update status"""
    try:
        # Check if it's a video file that needs audio extraction
        if is_video(file_path):
            transcription_status[task_id]['status'] = 'extracting_audio'
            audio_path = os.path.join('temp', f'audio_{task_id}.mp3')
            if not extract_audio(file_path, audio_path):
                transcription_status[task_id]['status'] = 'failed'
                return
        else:
            # For audio files, just copy to temp directory
            audio_path = os.path.join('temp', f'audio_{task_id}.mp3')
            shutil.copy2(file_path, audio_path)

        transcription_status[task_id]['status'] = 'transcribing'

        # Transcribe audio with language detection
        result = transcribe_audio(audio_path, translate=False)

        # Clean up
        os.remove(audio_path)
        os.remove(file_path)

        if result:
            transcription_status[task_id]['status'] = 'completed'
            transcription_status[task_id]['transcript'] = result["text"]
            transcription_status[task_id]['detected_language'] = result["language"]
            transcription_status[task_id]['language_confidence'] = round(
                result["language_probability"] * 100, 2)
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
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Generate unique task ID
        task_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(
            app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        file.save(file_path)

        # Initialize status
        transcription_status[task_id] = {
            'status': 'starting',
            'filename': filename
        }

        # Start processing in background
        thread = threading.Thread(
            target=process_file, args=(file_path, task_id))
        thread.start()

        return jsonify({
            'task_id': task_id,
            'message': 'File uploaded successfully'
        })

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/status/<task_id>')
def get_status(task_id):
    if task_id not in transcription_status:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(transcription_status[task_id])


if __name__ == '__main__':
    app.run(debug=True)
