import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai
from gtts import gTTS
import requests
import urllib.request
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from fer import FER
import cv2
from PIL import Image
import numpy as np

# ---------------------------
# 1. Load Environment Variables
# ---------------------------
load_dotenv(find_dotenv())

# ---------------------------
# 2. Set Up API Keys
# ---------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please set it in your .env file.")
    st.stop()

if not UNSPLASH_ACCESS_KEY:
    st.error("Unsplash Access Key not found. Please set it in your .env file.")
    st.stop()

# ---------------------------
# 3. Configure Google Generative AI
# ---------------------------
genai.configure(api_key=GOOGLE_API_KEY)

def gemini_pro():
    model = genai.GenerativeModel('gemini-pro')
    return model

# ---------------------------
# 4. Define Content Generation Function
# ---------------------------
def generate_content_based_on_emotion(emotion, user_choice, language):
    model = gemini_pro()
    
    # Define prompts for each content type
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
    
    prompt = prompts.get(user_choice)
    if not prompt:
        return "Invalid choice! Please select a valid content type."
    
    # Generate content using the selected model
    try:
        response = model.generate_content(prompt)
        
        # Extract the generated text from the response object
        if hasattr(response, 'candidates') and response.candidates:
            generated_text = response.candidates[0].content.parts[0].text
            return generated_text
        else:
            return "No content generated."
    except Exception as e:
        return f"An error occurred while generating content: {str(e)}"

# ---------------------------
# 5. Define Text-to-Speech Function
# ---------------------------
def text_to_speech(text, language='en'):
    try:
        tts = gTTS(text=text, lang=language)
        audio_path = "voiceover.mp3"
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        st.error(f"Text-to-Speech conversion failed: {str(e)}")
        return None

# ---------------------------
# 6. Define Image Fetching Function
# ---------------------------
def fetch_images(emotion, count=80):
    try:
        query = emotion
        url = f"https://api.unsplash.com/photos/random?query={query}&client_id={UNSPLASH_ACCESS_KEY}&count={count}"
        response = requests.get(url)
        if response.status_code == 200:
            images = [photo['urls']['regular'] for photo in response.json()]
            return images
        else:
            st.error("Failed to fetch images from Unsplash.")
            return None
    except Exception as e:
        st.error(f"Error fetching images: {str(e)}")
        return None

# ---------------------------
# 7. Define Image Download Function
# ---------------------------
def download_images(image_urls):
    image_paths = []
    try:
        for idx, url in enumerate(image_urls):
            image_path = f"image_{idx}.jpg"
            urllib.request.urlretrieve(url, image_path)
            image_paths.append(image_path)
        return image_paths
    except Exception as e:
        st.error(f"Error downloading images: {str(e)}")
        return None

# ---------------------------
# 8. Define Video Creation Function
# ---------------------------
def create_video(image_paths, audio_path, output_path='output_video.mp4', duration_per_image=5):
    try:
        clips = []
        for image in image_paths:
            clip = ImageClip(image).set_duration(duration_per_image)
            clips.append(clip)
        
        # Concatenate image clips
        video = concatenate_videoclips(clips, method="compose")
        
        # Add audio
        audio = AudioFileClip(audio_path)
        video = video.set_audio(audio)
        
        # Write the video file
        video.write_videofile(output_path, fps=24)
        return output_path
    except Exception as e:
        st.error(f"Error creating video: {str(e)}")
        return None

# ---------------------------
# 9. Define Emotion Detection Function
# ---------------------------
def detect_emotion(image):
    """
    Detects emotion from a given image.

    Parameters:
    - image (numpy.ndarray): The image in which to detect emotion.

    Returns:
    - emotion (str): The detected emotion with the highest score.
    """
    # Initialize the FER detector
    detector = FER(mtcnn=True)  # Using MTCNN for face detection

    # Detect emotions in the image
    emotions = detector.detect_emotions(image)

    if not emotions:
        return None  # No faces detected

    # Extract the emotions for each detected face
    detected_emotions = []
    for face in emotions:
        emotions_scores = face["emotions"]
        # Get the emotion with the highest score
        dominant_emotion = max(emotions_scores, key=emotions_scores.get)
        detected_emotions.append(dominant_emotion)

    # For simplicity, return the first detected emotion
    return detected_emotions[0] if detected_emotions else None

# ---------------------------
# 10. Streamlit UI
# ---------------------------
st.title("Emotion-Based Content Generator with Animated Video")

st.markdown("""
This application generates personalized content based on your current emotion and transforms it into an engaging animated video with voiceover.
""")

# **Emotion Input Options**
st.header("1. Provide Your Emotion")

emotion_option = st.radio(
    "Choose how you want to provide your emotion:",
    ("Type your emotion", "Capture emotion via camera")
)

if emotion_option == "Type your emotion":
    user_emotion = st.text_input("How are you feeling today?", placeholder="e.g., happy, anxious, calm")
elif emotion_option == "Capture emotion via camera":
    # Use camera input to capture image
    captured_image = st.camera_input("Capture your facial expression:")
    if captured_image:
        # Display the captured image
        image = Image.open(captured_image)
        st.image(image, caption='Captured Image', use_column_width=True)
        
        # Convert the image to a numpy array
        image_np = np.array(image.convert('RGB'))
        
        # Detect emotion
        with st.spinner("Detecting emotion from the captured image..."):
            detected_emotion = detect_emotion(image_np)
        
        if detected_emotion:
            st.success(f"Detected Emotion: **{detected_emotion.capitalize()}**")
            user_emotion = detected_emotion
        else:
            st.error("Could not detect any face or emotion in the captured image.")
            user_emotion = None
    else:
        user_emotion = None
else:
    user_emotion = None

# **Content Type Selection**
user_choice = st.selectbox("Select content type:", [
    "story", "cosmic", "short_story", "flash_fiction", 
    "poetry", "inspirational_quote", "journaling_prompt", 
    "graphic_novel", "art_storytelling", "cultural_story", 
    "interactive_fiction"
])

# Language Selection with Language Codes for gTTS
languages_display = [
    "English", "Spanish", "French", "German", "Chinese (Simplified)", 
    "Hindi", "Arabic", "Portuguese", "Russian", "Japanese", 
    "Korean", "Italian", "Dutch", "Bengali", "Tamil", "Turkish", 
    "Vietnamese", "Thai", "Greek", "Swedish", "Polish", 
    "Hebrew", "Indonesian", "Malay", "Persian", "Urdu"
]

language_codes = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Simplified)": "zh-CN",
    "Hindi": "hi",
    "Arabic": "ar",
    "Portuguese": "pt",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Italian": "it",
    "Dutch": "nl",
    "Bengali": "bn",
    "Tamil": "ta",
    "Turkish": "tr",
    "Vietnamese": "vi",
    "Thai": "th",
    "Greek": "el",
    "Swedish": "sv",
    "Polish": "pl",
    "Hebrew": "he",
    "Indonesian": "id",
    "Malay": "ms",
    "Persian": "fa",
    "Urdu": "ur"
}

language = st.selectbox("Choose a language for the content:", languages_display)
language_code = language_codes.get(language, "en")  # Default to English if not found

# **Generate Button**
if st.button("Generate"):
    with st.spinner("Generating content..."):
        if user_emotion and user_choice and language:
            # Step 1: Generate Content
            content = generate_content_based_on_emotion(user_emotion, user_choice, language)
            st.markdown(f"**Generated Content:**\n\n{content}")
            
            # Proceed only if content was successfully generated
            if content and not content.startswith("An error occurred") and not content.startswith("Invalid choice"):
                # Step 2: Generate Voiceover
                st.info("Generating voiceover...")
                audio_path = text_to_speech(content, language=language_code)
                if audio_path:
                    st.success("Voiceover generated.")
                    
                    # Step 3: Fetch Images
                    st.info("Fetching images...")
                    image_urls = fetch_images(user_emotion)
                    if image_urls:
                        # Step 4: Download Images
                        image_paths = download_images(image_urls)
                        if image_paths:
                            st.success("Images downloaded.")
                            
                            # Step 5: Create Video
                            st.info("Creating video...")
                            video_path = create_video(image_paths, audio_path)
                            if video_path:
                                st.success("Video created.")
                                
                                # Step 6: Display Video
                                video_file = open(video_path, "rb")
                                video_bytes = video_file.read()
                                st.video(video_bytes)
                                
                                # Step 7: Provide Download Link
                                st.download_button(
                                    label="Download Video",
                                    data=video_bytes,
                                    file_name="animated_content.mp4",
                                    mime="video/mp4"
                                )
                                
                                # Step 8: Clean Up Temporary Files
                                for img in image_paths:
                                    os.remove(img)
                                os.remove(audio_path)
                                os.remove(video_path)
                            else:
                                st.error("Video creation failed.")
                        else:
                            st.error("Image downloading failed.")
                    else:
                        st.error("Image fetching failed.")
        else:
            st.warning("Please provide your emotion (either type it or capture via camera), select a content type, and choose a language.")

# **Reset Button**
if st.button("Reset"):
    st.experimental_rerun()
