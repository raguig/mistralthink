import os

from dotenv import load_dotenv


load_dotenv()

api_key = (os.getenv("MISTRAL_API_KEY") or "").strip().strip('"').strip("'")
if not api_key:
    raise ValueError("MISTRAL_API_KEY not found in .env")

MODEL = "pixtral-large-latest"
CRITIC_MODEL = "mistral-small-latest"
SANDBOX_TIMEOUT_SECONDS = 12

