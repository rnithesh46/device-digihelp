import os
from dotenv import load_dotenv
from google import genai
from PIL import Image
import io

# Load environment variables from the .env file
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# ----------------------------------------------------
# 1. CORE AI SETUP
# ----------------------------------------------------

# Initialize the Gemini Client. It automatically uses the GEMINI_API_KEY environment variable.
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    client = None

# Optimized System Instruction for consistent output
SYSTEM_INSTRUCTION = (
    "You are an expert technical manual writer named 'DigiHelp'. Your goal is to "
    "generate clear, concise, and easy-to-follow step-by-step instructions for "
    "operating a device shown in the image. The user is a beginner. "
    "Always start the response by identifying the device first, and then give the steps."
    "The entire output must be formatted as a single, clean text string where the "
    "manual is a numbered list. DO NOT include any introductory or concluding sentences. "
    "Only provide the device name on the first line, and the numbered list starting on the next line."
)

# ----------------------------------------------------
# 2. CORE GENERATION FUNCTION
# ----------------------------------------------------

def generate_manual(image_path: str) -> tuple[str, str]:
    """
    Analyzes an image of a device using Gemini and generates a step-by-step manual.
    
    Args:
        image_path: Local file path to the uploaded device image.

    Returns:
        A tuple containing (device_name, text_manual)
    """
    if not client:
        raise ConnectionError("Gemini client not initialized. Check GEMINI_API_KEY.")

    try:
        # 1. Load the image using PIL (Pillow)
        img = Image.open(image_path)
        
        # 2. Define the user's specific request
        user_prompt = "Generate a step-by-step guide on 'How to use this device' for a beginner user. Ensure the steps are easy to understand."
        
        # 3. Configure the model for deterministic and focused output
        config = genai.types.GenerateContentConfig(
            # Setting a low temperature for less creative, more factual/deterministic responses
            temperature=0.1, 
            system_instruction=SYSTEM_INSTRUCTION
        )

        # 4. Call the Multimodal API
        response = client.models.generate_content(
            model='gemini-2.5-flash', # A fast and capable multimodal model
            contents=[img, user_prompt],
            config=config
        )

        # 5. Process the raw text output
        raw_text = response.text.strip()
        
        # The first line should be the device name, the rest is the manual
        lines = raw_text.split('\n', 1)
        
        # Assuming the model follows the instruction to put the name first
        device_name = lines[0].strip() if lines else "Unknown Device"
        text_manual = lines[1].strip() if len(lines) > 1 else raw_text
        
        # Clean up any remaining artifacts like initial number "1." from the model
        if text_manual.startswith("1. "):
             text_manual = text_manual[3:].strip()
        
        # Re-add the first step number to ensure a clean list starts
        text_manual = "1. " + text_manual.replace('\n', '\n')
        
        # Return the extracted components
        return device_name, text_manual

    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        # Re-raise a simpler error for the main application to catch
        raise RuntimeError(f"Failed to generate manual: {str(e)}")


# ----------------------------------------------------
# 3. Simple Test Block (Optional for local testing)
# ----------------------------------------------------
if __name__ == '__main__':
    # To run this test, you need to provide a dummy image file named 'test_device.jpg' 
    # in your root directory and ensure GEMINI_API_KEY is set in .env
    test_image_path = "test_device.jpg" 
    if os.path.exists(test_image_path):
        print(f"--- Running Test on: {test_image_path} ---")
        try:
            name, manual = generate_manual(test_image_path)
            print(f"Device: {name}")
            print("\n--- Generated Manual ---")
            print(manual)
            print("------------------------")
        except Exception as e:
            print(f"TEST FAILED: {e}")
    else:
        print(f"TEST SKIPPED: Place a file named '{test_image_path}' in the root directory to test.")