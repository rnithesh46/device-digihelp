import os
import io
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn
from pydantic import BaseModel, EmailStr
import smtplib
import ssl
from email.message import EmailMessage
from dotenv import load_dotenv
from typing import Optional 
from pathlib import Path
from googleapiclient.discovery import build 

# --- SIMPLIFIED .env LOADING ---
# This will load the .env file if it exists (on your computer)
# On Vercel, it will do nothing, and the Vercel environment variables will be used instead.
load_dotenv() 

# 1. Load Environment Variables
API_KEY = os.getenv("GEMINI_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_APP_PASSWORD = os.getenv("SENDER_APP_PASSWORD")
YOUR_NAME = "DeviceDigiHelp Support"
CUSTOM_SEARCH_API_KEY = os.getenv("CUSTOM_SEARCH_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

# 2. Validate Environment Variables
# We check this *after* load_dotenv()
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it in your .env file or Vercel settings.")
if not SENDER_EMAIL:
    raise ValueError("SENDER_EMAIL environment variable not set (your Gmail address). Please set it in your .env file or Vercel settings.")
if not SENDER_APP_PASSWORD:
    raise ValueError("SENDER_APP_PASSWORD environment variable not set (your 16-character App Password). Please set it in your .env file or Vercel settings.")
if not CUSTOM_SEARCH_API_KEY:
    raise ValueError("CUSTOM_SEARCH_API_KEY environment variable not set. Please set it in your .env file or Vercel settings.")
if not SEARCH_ENGINE_ID:
    raise ValueError("SEARCH_ENGINE_ID environment variable not set. Please set it in your .env file or Vercel settings.")

# 3. Configure Gemini
genai.configure(api_key=API_KEY)

# --- Prompts (Unchanged) ---
MANUAL_SYSTEM_PROMPT_TEMPLATE = """You are an expert tech assistant. A user has uploaded an image of a device.
You MUST generate your entire response in the following language: {language}.
Your response must be formatted in simple HTML.

1.  **Device Identification:** Start with a single line identifying the device. **Only bold the device name itself**. (e.g., "This appears to be an <b>Apple iPhone 14 Pro</b>.").
2.  **Quick Start Guide:**
    * Provide a clear, step-by-step 'Quick Start Guide' with the most essential, basic functions.
    * Focus on what a brand new user would need to know (e.g., How to turn on/off, main controls, core function).
    * Be easily understandable and to the point.
    * This should be a detailed list, including as many basic steps as needed.
3.  **Further Assistance:**
    * After the guide, add a 'Further Assistance' section.
    * Provide a list of 2-3 common follow-up questions.

RULES FOR FORMATTING:
- Use `<h3>` for main titles (like 'Quick Start Guide' and 'Further Assistance').
- Use `<h4>` for sub-titles (like 'Setup', 'Core Functions').
- Use `<ul>` and `<li>` for all bullet points and steps.
- Use `<b>` and `</b>` for all bold text.
- Do NOT use any markdown characters like '##', '###', '*', or '**'.
"""

TEXT_MANUAL_SYSTEM_PROMPT_TEMPLATE = """You are an expert tech assistant. A user has provided a device name.
You MUST generate your entire response in the following language: {language}.
Your response must be formatted in simple HTML.

Your task is to generate a comprehensive, accurate, and easy-to-understand step-by-step guide for a beginner, based on the user's query.

1.  **Device Identification:** Start with a single line confirming the device. **Only bold the device name itself**. (e.g., "Here is the guide for the <b>Apple iPhone 14 Pro</b>.").
2.  **Quick Start Guide:**
    * Provide a clear, step-by-step 'Quick Start Guide' with the most essential, basic functions.
    * Focus on what a brand new user would need to know (e.g., How to turn on/off, main controls, core function).
    * Be easily understandable and to the point.
    * This should be a detailed list, including as many basic steps as needed.
3.  **Further Assistance:**
    * After the guide, add a 'Further Assistance' section.
    * Provide a list of 2-3 common follow-up questions.

RULES FOR FORMATTING:
- Use `<h3>` for main titles (like 'Quick Start Guide' and 'Further Assistance').
- Use `<h4>` for sub-titles (like 'Setup', 'Core Functions').
- Use `<ul>` and `<li>` for all bullet points and steps.
- Use `<b>` and `</b>` for all bold text.
- Do NOT use any markdown characters like '##', '###', '*', or '**'.
"""

CHAT_SYSTEM_PROMPT_TEMPLATE = """You are a helpful, expert tech assistant. You are acting as a chatbot.
The user has already identified a device, which will be provided as 'Device Context'.
You MUST generate your entire response in the following language: {language}.
Your job is to answer the user's follow-up questions with detailed, accurate, and step-by-step instructions.

- Strict scope: only answer questions directly about the provided device context. If the user asks anything unrelated (general chit-chat, unrelated topics, other devices, or personal questions), respond with a brief refusal like: "I can help only with questions about this device."

- If the user provides an image with their question, use it as additional context (e.g., if they ask 'what is this button?' and provide an image, you must identify the button in the image).
- If no image is provided, just answer the text question.
- You must be able to answer any question, from simple to complex (e.g., "How do I add a fingerprint?", "How much detergent do I put in this washing machine?", "How do I print multiple copies?").
- Provide clear, concise, and helpful answers.
- Format your response with simple HTML (<b>, <ul>, <li>) for clarity.
"""

try:
    generation_config = genai.GenerationConfig(
        temperature=0.7,
        top_p=1,
        top_k=1,
        max_output_tokens=8192,
    )
except Exception as e:
    print(f"Error configuring generation: {e}")

# 4. Set up the FastAPI App
app = FastAPI(title="Device DigiHelp API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Form Data ---
class ContactForm(BaseModel):
    name: str
    email: EmailStr
    message: str

class TextManualRequest(BaseModel):
    query: str
    language: str

# Note: We no longer use a Pydantic model for the chat, as it now uses FormData

# --- Helper Function for Sending Email ---
def send_email(subject: str, body: str, reply_to: EmailStr = None):
    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = f"{YOUR_NAME} <{SENDER_EMAIL}>"
        msg['To'] = SENDER_EMAIL
        if reply_to:
            msg['Reply-To'] = reply_to
        msg.set_content(body)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
            server.send_message(msg)
        print(f"Successfully sent email: {subject}")
        return True
    except Exception as e:
        print(f"--- EMAIL SENDING FAILED ---")
        print(f"Error: {e}")
        return False

# --- Helper Function for Google Image Search ---
def get_image_url(query: str) -> Optional[str]:
    try:
        service = build("customsearch", "v1", developerKey=CUSTOM_SEARCH_API_KEY)
        result = service.cse().list(
            q=query,
            cx=SEARCH_ENGINE_ID,
            searchType="image",
            num=1,
            safe="high" # Enable safe search
        ).execute()
        
        if "items" in result and len(result["items"]) > 0:
            return result["items"][0]["link"]
        else:
            return None
    except Exception as e:
        print(f"Error during image search: {e}")
        return None

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Device DigiHelp AI Backend is running."}

@app.post("/generate-manual/")
async def generate_manual(
    language: str = Form("English"),
    file: UploadFile = File(...)
):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))
        dynamic_system_prompt = MANUAL_SYSTEM_PROMPT_TEMPLATE.format(language=language)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        instructed_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-preview-09-2025",
            system_instruction=dynamic_system_prompt
        )
        user_prompt_text = "Please identify this device and generate its manual."
        response = instructed_model.generate_content(
            [user_prompt_text, img],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False,
        )
        if response and response.text:
            return {"manual_text": response.text, "image_url": None} # No image URL needed here
        else:
            raise HTTPException(status_code=500, detail="Failed to generate content or response was blocked.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# --- Generate Manual from Text Endpoint ---
@app.post("/generate-manual-from-text/")
async def generate_manual_from_text(request: TextManualRequest):
    try:
        image_url = get_image_url(request.query) 
        
        dynamic_system_prompt = TEXT_MANUAL_SYSTEM_PROMPT_TEMPLATE.format(language=request.language)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        instructed_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-preview-09-2025",
            system_instruction=dynamic_system_prompt
        )
        response = instructed_model.generate_content(
            [request.query],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False,
        )

        if response and response.text:
            return {"manual_text": response.text, "image_url": image_url}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate content or response was blocked.")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# --- AI Chatbot Endpoint (NOW MULTIMODAL) ---
@app.post("/ask-follow-up/")
async def ask_follow_up(
    device: str = Form(...),
    question: str = Form(...),
    language: str = Form("English"),
    file: Optional[UploadFile] = File(None) # Optional image file
):
    
    try:
        dynamic_chat_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(language=language)
        chat_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-preview-09-2025",
            system_instruction=dynamic_chat_prompt
        )
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        prompt_parts = []
        prompt_parts.append(f"Device Context: {device}")
        
        if file:
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Invalid file type for chat. Please upload an image.")
            image_bytes = await file.read()
            img = Image.open(io.BytesIO(image_bytes))
            prompt_parts.append("Here is an image related to my question:")
            prompt_parts.append(img)
        
        prompt_parts.append(f"User Question: {question}")
        
        response = chat_model.generate_content(
            prompt_parts,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        if response and response.text:
            return {"answer": response.text}
        else:
            raise HTTPException(status_code=500, detail="AI failed to generate a chat response.")
            
    except Exception as e:
        print(f"Error in chat follow-up: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# --- Contact Form Submit Endpoint ---
@app.post("/contact-submit/")
async def submit_contact_form(form_data: ContactForm):
    try:
        print(f"--- NEW CONTACT FORM SUBMISSION ---")
        print(f"Name: {form_data.name}")
        print(f"Email: {form_data.email}")
        print(f"Message: {form_data.message}")
        subject = f"New Contact Form from {form_data.name}"
        body = f"""
        You received a new contact form submission:
        
        Name: {form_data.name}
        Email: {form_data.email}
        
        Message:
        {form_data.message}
        """
        send_email(subject, body, reply_to=form_data.email)
        print(f"---------------------------------")
        return {"status": "success", "message": "Contact form received!"}
    except Exception as e:
        print(f"Error in contact form: {e}")
        raise HTTPException(status_code=500, detail="Error processing contact form.")

# 5. Run the App
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)