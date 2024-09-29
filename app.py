import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from gtts import gTTS
import requests
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import urllib.request

# Load environment variables
load_dotenv()

# Set Up API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please set it in your .env file.")
    st.stop()

if not UNSPLASH_ACCESS_KEY:
    st.error("Unsplash Access Key not found. Please set it in your .env file.")
    st.stop()

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

def get_gemini_pro_model():
    return genai.GenerativeModel('gemini-pro')

# Content Generation Function
def generate_content(emotion, content_type, language):
    model = get_gemini_pro_model()
    prompts = {
        'story': f"Tell me a healing story that reflects the feeling of {emotion} and helps the listener overcome it. Please provide the story in {language}.",
        'cosmic': f"Share a cosmic tale that resonates with the feeling of {emotion} and offers a perspective to rise above it. Please provide the tale in {language}.",
        'short_story': f"Write a short story that captures the feeling of {emotion}. Please provide it in {language}.",
        'flash_fiction': f"Create a flash fiction piece that embodies the emotion of {emotion}. Please provide it in {language}.",
        'poetry': f"Compose a poem that reflects the feeling of {emotion}. Please provide it in {language}.",
        'inspirational_quote': f"Share an inspirational quote that relates to the emotion of {emotion}. Please provide it in {language}.",
        'journaling_prompt': f"Suggest a journaling prompt for someone feeling {emotion}. Please provide it in {language}.",
        'graphic_novel': f"Outline a graphic novel concept that involves the emotion of {emotion}. Please provide it in {language}.",
        'art_storytelling': f"Describe an art piece that tells a story about {emotion}. Please provide it in {language}.",
        'cultural_story': f"Tell a cultural story that reflects the emotion of {emotion}. Please provide it in {language}.",
        'interactive_fiction': f"Create an interactive fiction scenario based on the emotion of {emotion}. Please provide it in {language}."
    }

    prompt = prompts.get(content_type)
    if not prompt:
        return "Invalid content type selected."

    try:
        response = model.generate_content(prompt)
        if hasattr(response, 'candidates') and response.candidates:
            return response.candidates[0].content.parts[0].text
        return "No content generated."
    except Exception as e:
        return f"Error generating content: {e}"

# Text-to-Speech Function
def text_to_speech(text, language='en'):
    try:
        tts = gTTS(text=text, lang=language)
        audio_path = "voiceover.mp3"
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        st.error(f"Text-to-Speech conversion failed: {e}")
        return None

# Image Fetching Function
def fetch_images(emotion, count=50):    # images will come sequentially at max. of 3 min. Need to work on it.
    try:
        url = f"https://api.unsplash.com/photos/random?query={emotion}&client_id={UNSPLASH_ACCESS_KEY}&count={count}"
        response = requests.get(url)
        response.raise_for_status()
        return [photo['urls']['regular'] for photo in response.json()]
    except Exception as e:
        st.error(f"Error fetching images: {e}")
        return None

# Image Download Function
def download_images(image_urls):
    image_paths = []
    try:
        for idx, url in enumerate(image_urls):
            image_path = f"image_{idx}.jpg"
            urllib.request.urlretrieve(url, image_path)
            image_paths.append(image_path)
        return image_paths
    except Exception as e:
        st.error(f"Error downloading images: {e}")
        return None

# Video Creation Function
def create_video(image_paths, audio_path, output_path='output_video.mp4', duration_per_image=5):
    try:
        clips = [ImageClip(img).set_duration(duration_per_image) for img in image_paths]
        video = concatenate_videoclips(clips, method="compose")
        audio = AudioFileClip(audio_path)
        video = video.set_audio(audio)
        video.write_videofile(output_path, fps=24)
        return output_path
    except Exception as e:
        st.error(f"Error creating video: {e}")
        return None

# Emotion Detection Function
def detect_emotion(image):
    try:
        img = Image.open(image).convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        analysis = DeepFace.analyze(img_cv, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list) and analysis:
            return analysis[0]['dominant_emotion']
        elif isinstance(analysis, dict):
            return analysis['dominant_emotion']
        return None
    except Exception as e:
        st.error(f"Emotion detection failed: {e}")
        return None

# Streamlit UI
st.title("Emotion-Based Content Generator with Animated Video")
st.markdown("""
Generate personalized content based on your current emotion and transform it into an engaging animated video with voiceover.
""")

# Emotion Input
st.header("Provide Your Emotion")
emotion_input_method = st.radio("Choose how to provide your emotion:", ("Manual Input", "Use Camera"))
user_emotion = None

if emotion_input_method == "Manual Input":
    user_emotion = st.text_input("How are you feeling today?", placeholder="e.g., happy, anxious, calm")
elif emotion_input_method == "Use Camera":
    captured_image = st.camera_input("Take a picture to detect your emotion")
    if captured_image:
        st.image(captured_image, caption="Captured Image", use_column_width=True)
        with st.spinner("Detecting emotion..."):
            detected_emotion = detect_emotion(captured_image)
            if detected_emotion:
                st.success(f"Detected Emotion: {detected_emotion}")
                user_emotion = detected_emotion
            else:
                st.error("Failed to detect emotion.")

# Content Type Selection
content_types = [
    "story", "cosmic", "short_story", "flash_fiction", 
    "poetry", "inspirational_quote", "journaling_prompt", 
    "graphic_novel", "art_storytelling", "cultural_story", 
    "interactive_fiction"
]
user_choice = st.selectbox("Select content type:", content_types)

# Language Selection
languages = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Chinese (Simplified)": "zh-CN", "Hindi": "hi", "Arabic": "ar",
    "Portuguese": "pt", "Russian": "ru", "Japanese": "ja",
    "Korean": "ko", "Italian": "it", "Dutch": "nl", "Bengali": "bn",
    "Tamil": "ta", "Turkish": "tr", "Vietnamese": "vi", "Thai": "th",
    "Greek": "el", "Swedish": "sv", "Polish": "pl", "Hebrew": "he",
    "Indonesian": "id", "Malay": "ms", "Persian": "fa", "Urdu": "ur"
}
language = st.selectbox("Choose a language for the content:", list(languages.keys()))
language_code = languages.get(language, "en")

# Generate Button
if st.button("Generate"):
    if user_emotion and user_choice and language:
        with st.spinner("Generating content..."):
            # Generate Content
            content = generate_content(user_emotion, user_choice, language)
            st.markdown(f"**Generated Content:**\n\n{content}")

            if content and not content.startswith("Error") and not content.startswith("Invalid"):
                # Generate Voiceover
                st.info("Generating voiceover...")
                audio_path = text_to_speech(content, language=language_code)
                if audio_path:
                    st.success("Voiceover generated.")
                    
                    # Fetch Images
                    st.info("Fetching images...")
                    image_urls = fetch_images(user_emotion)
                    if image_urls:
                        # Download Images
                        image_paths = download_images(image_urls)
                        if image_paths:
                            st.success("Images downloaded.")
                            
                            # Create Video
                            st.info("Creating video...")
                            video_path = create_video(image_paths, audio_path)
                            if video_path:
                                st.success("Video created.")
                                
                                # Display Video
                                with open(video_path, "rb") as video_file:
                                    video_bytes = video_file.read()
                                    st.video(video_bytes)
                                
                                # Download Link
                                st.download_button(
                                    label="Download Video",
                                    data=video_bytes,
                                    file_name="animated_content.mp4",
                                    mime="video/mp4"
                                )
                            
                            # Clean Up
                            for img_path in image_paths:
                                if os.path.exists(img_path):
                                    os.remove(img_path)
                            if os.path.exists(audio_path):
                                os.remove(audio_path)
                            if os.path.exists(video_path):
                                os.remove(video_path)
                else:
                    st.error("Voiceover generation failed.")
            else:
                st.error("Content generation failed.")
    else:
        st.error("Please fill all fields before generating.")
