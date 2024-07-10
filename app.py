import streamlit as st
import joblib
import librosa
import numpy as np
import os

# Load model
filename = "./SVM.sav"
model = joblib.load(filename)

def extract_features(data,sr=22050,frame_length=2048,hop_length=512):
    result=np.array([])
    mfcc=np.mean(librosa.feature.mfcc(y=data,sr=sr).T, axis=0)
    result=np.hstack((result,mfcc))
    return result

def get_features(path,duration=2.5, offset=0.6):
    data,sr=librosa.load(path,duration=duration,offset=offset)
    aud=extract_features(data)
    audio=np.array(aud)
    return audio

def predict_emotion(audio_path):
    X = []
    feature = get_features(audio_path)
    X.append(feature)
    pred = model.predict(X)
    return pred
    

st.title('Speech Emotion Detection')

uploaded_audio = st.file_uploader('Upload audio', type=['wav', 'mp3'], accept_multiple_files=True)

if uploaded_audio:
    st.write('Prediction:')
    for audio in uploaded_audio:
        st.write(audio.name)
        st.audio(audio, format='audio/wav')
        with open("temp_audio.wav", "wb") as f:
            f.write(audio.getbuffer())
        pred = predict_emotion("temp_audio.wav")

        if pred == 'angry':
            st.write("Model detects Angry")
        elif pred == 'happy':
            st.write("Model detects Happy")
        elif pred == 'sad':
            st.write("Model detects Sad")
        else:
            st.write("Model can't detect the emotion")

        os.remove("temp_audio.wav")

