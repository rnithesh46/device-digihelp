import os
import io
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn
from pydantic import BaseModel, EmailStr
import smtplib
import ssl
from email.message import EmailMessage

# 1. Load Environment Variables
API_KEY = os.getenv("GEMINI_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_APP_PASSWORD = os.getenv("SENDER_APP_PASSWORD")
YOUR_NAME = "DeviceDigiHelp Support"

# 2. Validate Environment Variables
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
if not SENDER_EMAIL:
    raise ValueError("SENDER_EMAIL environment variable not set (your Gmail address).")
if not SENDER_APP_PASSWORD:
    raise ValueError("SENDER_APP_PASSWORD environment variable not set (your 16-character App Password).")

# 3. Configure Gemini
genai.configure(api_key=API_KEY)

# *** UPDATED PROMPT: Re-added language template ***
SYSTEM_PROMPT_TEMPLATE = """You are an expert tech assistant. A user has uploaded an image of a device.
You MUST generate your entire response in the following language: {language}.
Your response must be formatted in simple HTML.

1.  **Device Identification:** Start by identifying the device (e.g., "This appears to be an Apple iPhone 14 Pro.").
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
- Use `<ul>` and `<li>` for all bullet points and steps.
- Use `<b>` and `</b>` for all bold text.
- Do NOT use any markdown characters like '##', '###', '*', or '**'.
"""

try:
    # We will create the model instance dynamically per request to set the system instruction
    generation_config = genai.GenerationConfig(
        temperature=0.7,
        top_p=1,
        top_k=1,
        max_output_tokens=8192,
    )
except Exception as e:
    print(f"Error configuring generative model: {e}")

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

class ChatMessage(BaseModel):
    message: str

class ContactForm(BaseModel):
    name: str
    email: EmailStr
    message: str

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
        print("Please check your SENDER_EMAIL and SENDER_APP_PASSWORD.")
        print("------------------------------")
        return False

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Device DigiHelp AI Backend is running."}

@app.post("/generate-manual/")
async def generate_manual(
    language: str = Form("English"), # NEW: Re-added language parameter
    file: UploadFile = File(...)
):
    
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))

        # NEW: Create the dynamic system prompt
        dynamic_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(language=language)

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # Re-creating the model with new instructions for each call
        instructed_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-preview-09-2025",
            system_instruction=dynamic_system_prompt
        )
        
        response = instructed_model.generate_content(
            [img], # Send just the image
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False,
        )

        if response and response.text:
            return {"manual_text": response.text}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate content or response was blocked.")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# --- Chat Submit Endpoint ---
@app.post("/chat-submit/")
async def submit_chat(chat_message: ChatMessage):
    try:
        print(f"--- NEW CHAT MESSAGE ---")
        print(f"Message: {chat_message.message}")
        subject = "New Chat Message from DeviceDigiHelp"
        body = f"A user sent a new chat message:\n\n\"{chat_message.message}\""
        send_email(subject, body)
        print(f"--------------------------")
        return {"status": "success", "message": "Chat message received!"}
    except Exception as e:
        print(f"Error in chat submit: {e}")
        raise HTTPException(status_code=500, detail="Error processing chat message.")

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

