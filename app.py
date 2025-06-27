from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
import os
import torch
import torchaudio
from train import SpeakerModel, AudioDataset
import librosa
import numpy as np
from werkzeug.utils import secure_filename
import cv2
import speech_recognition as sr
import time
from datetime import datetime
import soundfile as sf
from threading import Thread
import random

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['AUDIO_FOLDER'] = 'static/audio'
app.config['VIDEO_FOLDER'] = 'static/videos'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)
os.makedirs(app.config['VIDEO_FOLDER'], exist_ok=True)

# Global variables for recording
is_recording = False
is_paused = False
frames = []
audio_frames = []
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Interview questions
questions = {
    "personal": [
        "Tell me about yourself.",
        "What are your strengths and weaknesses?",
        "Where do you see yourself in 5 years?",
        "Why should we hire you?",
        "What motivates you?"
    ],
    "technical": [
        "Explain your approach to problem-solving.",
        "Describe a challenging project you worked on.",
        "What programming languages are you proficient in?",
        "How do you stay updated with industry trends?",
        "Explain a technical concept to a non-technical person."
    ],
    "situational": [
        "Describe a time you faced a conflict at work and how you handled it.",
        "Tell me about a time you failed and what you learned.",
        "How would you handle a tight deadline with multiple priorities?",
        "Describe a time you had to work with a difficult team member.",
        "Give an example of when you showed leadership."
    ]
}

# Initialize speaker recognition model
def init_speaker_model():
    try:
        # Load model with fixed 24 output classes to match saved weights
        model = SpeakerModel(24)  # Hardcoded to match the saved model
        
        # Load weights (map to CPU if needed)
        state_dict = torch.load("models/speaker_recognition_model.pth", 
                              map_location=torch.device('cpu'))
        
        # Handle potential mismatch in state_dict keys
        if all(key.startswith('module.') for key in state_dict.keys()):
            # Remove 'module.' prefix if model was saved as DataParallel
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.eval()
        
        # Initialize label encoder (create dummy one if needed)
        dataset = AudioDataset(r"D:\AI mock interview")
        return model, dataset.label_encoder
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        # Fallback to create empty model if loading fails
        model = SpeakerModel(24)
        model.eval()
        return model, None

speaker_model, label_encoder = init_speaker_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/create_profile', methods=['GET', 'POST'])
def create_profile():
    if request.method == 'POST':
        session['name'] = request.form['name']
        session['email'] = request.form['email']
        session['password'] = request.form['password']
        session['field'] = request.form['field']
        return redirect(url_for('quiz'))
    return render_template('create_profile.html')

@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    if request.method == 'POST':
        session['quiz_answers'] = {
            'q1': request.form['q1'],
            'q2': request.form['q2'],
            'q3': request.form['q3']
        }
        return redirect(url_for('interview'))
    
    quiz_questions = [
        "What is your experience level?",
        "What type of job are you preparing for?",
        "What are your main areas you want to practice?"
    ]
    return render_template('quiz.html', questions=quiz_questions)

@app.route('/interview')
def interview():
    if 'name' not in session:
        return redirect(url_for('create_profile'))
    return render_template('interview.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global is_recording, frames, audio_frames
    is_recording = True
    frames = []
    audio_frames = []
    
    Thread(target=record_video).start()
    Thread(target=record_audio).start()
    
    return jsonify({'status': 'Recording started'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global is_recording
    is_recording = False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"interview_{timestamp}.mp4"
    audio_filename = f"interview_{timestamp}.wav"
    transcript_filename = f"interview_{timestamp}.txt"
    
    # Save video
    video_path = os.path.join(app.config['VIDEO_FOLDER'], video_filename)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))
    for frame in frames:
        out.write(frame)
    out.release()
    
    # Save audio
    audio_path = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
    sf.write(audio_path, np.concatenate(audio_frames), 16000)
    
    # Save transcript
    transcript_path = os.path.join(app.config['UPLOAD_FOLDER'], transcript_filename)
    with open(transcript_path, 'w') as f:
        f.write("Interview transcript will be here after processing.")
    
    session['last_recording'] = {
        'video': video_filename,
        'audio': audio_filename,
        'transcript': transcript_filename
    }
    
    return jsonify({
        'status': 'Recording stopped',
        'video': video_filename,
        'audio': audio_filename
    })

@app.route('/save_video', methods=['POST'])
def save_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"interview_{timestamp}.webm"
    video_path = os.path.join(app.config['VIDEO_FOLDER'], filename)
    video_file.save(video_path)
    
    return jsonify({'success': True, 'filename': filename})

def record_video():
    global is_recording, frames
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while is_recording:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            frames.append(frame)
        time.sleep(0.05)
    
    cap.release()

def record_audio():
    global is_recording, audio_frames
    
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        
        while is_recording:
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=5)
                audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                audio_frames.append(audio_data)
            except sr.WaitTimeoutError:
                continue

if __name__ == '__main__':
    app.run(debug=True)