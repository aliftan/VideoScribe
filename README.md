# VideoScribe

VideoScribe is a web application that automatically transcribes speech from video files into text using OpenAI's Whisper model.

## Features

- Upload video files (supports MP4, AVI, MOV, WMV, MKV)
- Real-time upload progress tracking
- Automatic speech-to-text transcription
- Processing status updates
- Clean, modern user interface

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/VideoScribe.git
cd VideoScribe
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:

```bash
python app.py
```

2. Open your browser and navigate to:

```
http://localhost:5000
```

3. Upload a video file and wait for the transcription to complete.

## Technologies Used

- Backend: Flask (Python)
- Frontend: Vue.js, Tailwind CSS
- Speech Recognition: OpenAI Whisper
- Video Processing: MoviePy

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
