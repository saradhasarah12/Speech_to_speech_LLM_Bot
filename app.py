import streamlit as st
import speech_recognition as sr
from transformers import pipeline
from gtts import gTTS
from pydub import AudioSegment
import tempfile
import os

# Initialize components
recognizer = sr.Recognizer()
llm = pipeline("text-generation", model="gpt2")

# Streamlit app title
st.title("Speech-to-Speech LLM Model")

# Function to convert speech to text
def process_audio():
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
    
    try:
        # Convert speech to text
        text = recognizer.recognize_google(audio)
        st.write(f"**You said:** {text}")
        
        # Generate response using LLM
        response = llm(text, max_length=50)[0]['generated_text']
        st.write(f"**Response:** {response}")
        
        # Convert response to MP3 and save it as an audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_audio_file_path = temp_audio_file.name
            tts = gTTS(text=response, lang='en')
            tts.save(temp_audio_file_path)
            
            # Play the audio file
            st.audio(temp_audio_file_path, format='audio/mp3')
            
            # Clean up the temporary file
            os.remove(temp_audio_file_path)
        
    except sr.UnknownValueError:
        st.write("Could not understand the audio.")
    

# Streamlit button to start speech recognition
if st.button("Start Listening"):
    process_audio()

# Instructions for user
st.write("Press the 'Start Listening' button and start speaking. The app will listen, generate a response using an LLM, display it, and speak it aloud.")
