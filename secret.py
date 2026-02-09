from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_KEY = os.environ.get("GEMINI_KEY")
OPENAI_KEY = os.environ.get("OPENAI_KEY")
FMP_KEY = os.environ.get("FMP_KEY")