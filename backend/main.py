import os
import io
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn
from pydantic import BaseModel, EmailStr # Added for form data
import smtplib # Added for email
import ssl # Added for email
from email.message import EmailMessage # Added for email

# 1. Load Environment Variables
# These MUST be set in your terminal before running
API_KEY = os.getenv("GEMINI_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_APP_PASSWORD = os.getenv("SENDER_APP_PASSWORD")
YOUR_NAME = "DeviceDigiHelp Support" # This will be the "From" name on the email

# 2. Validate Environment Variables
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
if not SENDER_EMAIL:
    raise ValueError("SENDER_EMAIL environment variable not set (your Gmail address).")
if not SENDER_APP_PASSWORD:
    raise ValueError("SENDER_APP_PASSWORD environment variable not set (your 16-character App Password).")

# 3. Configure Gemini
genai.configure(api_key=API_KEY)

SYSTEM_PROMPT = """You are an expert tech assistant. A user has uploaded an image of a device.
1.  Identify the device in the image. Be as specific as possible (e.g., "This appears to be a Sony PlayStation 5 controller").
2.  Generate a **very brief 'Quick Start Guide'** (5 bullet points MAX) for its most common functions.
3.  After the guide, add a new section titled "## Further Assistance".
4.  Under this new section, provide a list of 2-3 common follow-up questions a user might have about *this specific device*.
5.  Format the entire output clearly using Markdown (e.g., ## Headers, ### Sub-Headers, and * Bullet points).
"""

try:
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-preview-09-2025",
        system_instruction=SYSTEM_PROMPT
    )
    generation_config = genai.GenerationConfig(
        temperature=0.7,
        top_p=1,
        top_k=1,
        max_output_tokens=8192,
    )
except Exception as e:
    print(f"Error configuring generative model: {e}")
    model = None

# 4. Set up the FastAPI App
app = FastAPI(title="Device DigiHelp API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
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
    """
    Sends an email using the pre-configured SENDER_EMAIL.
    """
    try:
        # Create the email message object
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = f"{YOUR_NAME} <{SENDER_EMAIL}>"
        msg['To'] = SENDER_EMAIL  # Send the email to yourself
        
        # If a 'reply_to' is provided (from the contact form), add it
        # This makes hitting "Reply" in Gmail work correctly
        if reply_to:
            msg['Reply-To'] = reply_to
            
        msg.set_content(body)

        # Create a secure SSL context
        context = ssl.create_default_context()

        # Log in to Gmail and send the email
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
async def generate_manual(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Generative model is not configured correctly.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        response = model.generate_content(
            [img],
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
        
        # --- Send Email Logic ---
        subject = "New Chat Message from DeviceDigiHelp"
        body = f"""
        A user sent a new chat message:
        
        "{chat_message.message}"
        """
        send_email(subject, body)
        # --- End Email Logic ---
        
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
        
        # --- Send Email Logic ---
        subject = f"New Contact Form from {form_data.name}"
        body = f"""
        You received a new contact form submission:
        
        Name: {form_data.name}
        Email: {form_data.email}
        
        Message:
        {form_data.message}
        """
        # We pass the user's email as the 'reply_to' address
        send_email(subject, body, reply_to=form_data.email)
        # --- End Email Logic ---

        print(f"---------------------------------")
        return {"status": "success", "message": "Contact form received!"}
    except Exception as e:
        print(f"Error in contact form: {e}")
        raise HTTPException(status_code=500, detail="Error processing contact form.")

# 5. Run the App
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)