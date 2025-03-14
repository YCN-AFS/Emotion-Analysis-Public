import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Model Configuration
GPT_MODEL = "gpt-3.5-turbo"
WHISPER_MODEL = "whisper-1"

# Supported Audio Formats
SUPPORTED_AUDIO_FORMATS = ['mp3', 'mp4', 'm4a', 'wav', 'webm']

# Language Configuration
DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = {
    "en": "English",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "ru": "Russian",
    "th": "Thai",
    "id": "Indonesian"
} 