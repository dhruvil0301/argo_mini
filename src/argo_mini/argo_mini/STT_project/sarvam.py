from dotenv import load_dotenv
from sarvamai import SarvamAI
import os

load_dotenv()

client = SarvamAI(
    api_subscription_key=os.getenv("SARVAM_API_KEY")
)

with open("test2.wav", "rb") as audio_file:
    response = client.speech_to_text.transcribe(
        file=audio_file,
        language_code="unknown"
    )

print("\n===== Speech-to-Text Result =====")
print("Request ID :", response.request_id)
print("Language   :", response.language_code)
print("Transcript :", response.transcript)
print("=================================")