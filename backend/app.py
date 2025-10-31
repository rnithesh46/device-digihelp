import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # To allow the frontend (on a different port) to talk to the backend
from dotenv import load_dotenv

# Import the core logic functions
from api_services.gemini_core import generate_manual
from api_services.tts_core import generate_audio

# ----------------------------------------------------
# 1. SETUP AND CONFIGURATION
# ----------------------------------------------------

# Load environment variables (like API keys) from .env file
load_dotenv()

app = Flask(__name__)
# Enable CORS for development so the frontend can connect
CORS(app) 

# Define file storage paths
# The path to temporarily store uploaded images
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
# The path where generated audio files are saved
MANUALS_FOLDER = os.path.join(os.path.dirname(__file__), 'manuals')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MANUALS_FOLDER'] = MANUALS_FOLDER

# Ensure the directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MANUALS_FOLDER, exist_ok=True)


# ----------------------------------------------------
# 2. MAIN API ROUTE: PROCESS IMAGE
# ----------------------------------------------------

@app.route('/api/process_device_image', methods=['POST'])
def process_manual_request():
    """
    Handles the image upload, calls the Gemini model, and generates the audio file.
    """
    # 1. Input Validation and File Handling
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    # Save the file to the temporary uploads directory
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    
    try:
        image_file.save(image_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500
    
    device_name = "Unknown Device"
    text_manual = "Error: Manual could not be generated."
    audio_file_url = None
    
    try:
        # 2. CORE LOGIC: Generate Text Manual (Gemini API)
        print(f"--> Calling Gemini to analyze: {image_path}")
        device_name, text_manual = generate_manual(image_path)
        print(f"--> Gemini identified: {device_name}")

        # 3. CORE LOGIC: Generate Audio Manual (TTS API)
        # We only call TTS if the text manual was successfully generated
        if text_manual and text_manual.strip():
            audio_filename = generate_audio(text_manual)
            if audio_filename:
                # Construct the URL for the frontend to download the audio
                audio_file_url = f"/api/download/{audio_filename}"

    except Exception as e:
        print(f"A major error occurred during processing: {e}")
        # Return a server error, but try to clean up
        return jsonify({"error": f"Processing failed due to a server error: {str(e)}"}), 500
    
    finally:
        # 4. Cleanup and Response
        # IMPORTANT: Remove the uploaded image file after processing
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"--> Cleaned up uploaded file: {image_path}")
        
    # Return the final structured data to the frontend
    return jsonify({
        "success": True,
        "device_identified": device_name,
        "text_manual": text_manual,
        "audio_file_url": audio_file_url # Will be null if TTS failed
    })

# ----------------------------------------------------
# 3. DOWNLOAD ROUTE: SERVE AUDIO FILE
# ----------------------------------------------------

@app.route('/api/download/<filename>')
def download_file(filename):
    """
    Serves the generated audio file to the frontend via its URL.
    """
    # Send the file from the manuals folder
    return send_from_directory(app.config['MANUALS_FOLDER'], filename)


# ----------------------------------------------------
# 4. RUN THE APPLICATION
# ----------------------------------------------------

if __name__ == '__main__':
    # Flask runs on port 5000 by default (the frontend will use another port, e.g., 3000)
    app.run(debug=True, port=5000)