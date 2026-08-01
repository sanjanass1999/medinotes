import os
from fastapi import FastAPI, Depends  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore
from pydantic import BaseModel  # type: ignore
from fastapi_clerk_auth import ClerkConfig, ClerkHTTPBearer, HTTPAuthorizationCredentials  # type: ignore
from google import genai  # type: ignore
from google.genai import types  # <-- Import types for native Gemini configuration
from dotenv import load_dotenv  # <-- 1. Import dotenv

# 2. Securely load your local environment variables
load_dotenv(dotenv_path=".env.local")

app = FastAPI()

# Configuration for Clerk verification
clerk_config = ClerkConfig(jwks_url=os.getenv("CLERK_JWKS_URL"))
clerk_guard = ClerkHTTPBearer(clerk_config)


class Visit(BaseModel):
    patient_name: str
    date_of_visit: str
    notes: str


# System instructions explicitly defining formatting constraints
system_prompt = """
You are provided with notes written by a doctor from a patient's visit.
Your job is to summarize the visit for the doctor and provide an email.
Reply with exactly three sections with the headings:
### Summary of visit for the doctor's records
### Next steps for the doctor
### Draft of email to patient in patient-friendly language
"""


def user_prompt_for(visit: Visit) -> str:
    return f"""Create the summary, next steps and draft email for:
Patient Name: {visit.patient_name}
Date of Visit: {visit.date_of_visit}
Notes:
{visit.notes}"""


@app.post("/api")
def consultation_summary(
    visit: Visit,
    creds: HTTPAuthorizationCredentials = Depends(clerk_guard),
):
    user_id = creds.decoded["sub"]  # Available for tracking/auditing
    
    # Construct clean user data context payload
    user_prompt = user_prompt_for(visit)

    def event_stream():
        # Keep the client scope tightly bound to the streaming threadpool context
        with genai.Client() as client:
            # Native Gemini setup passing the system instructions configuration parameter
            stream = client.models.generate_content_stream(
                model='gemini-2.5-flash',
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt
                )
            )
            
            for chunk in stream:
                text = chunk.text
                if text:
                    # Escape raw newlines to keep the SSE network packets perfectly continuous
                    clean_text = text.replace("\n", "\\n")
                    yield f"data: {clean_text}\n\n"

    # Disable Vercel proxy response buffering to get that instant typing effect
    return StreamingResponse(
        event_stream(), 
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"}
    )