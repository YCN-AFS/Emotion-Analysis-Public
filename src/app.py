import streamlit as st
import json
from emotion_analysis import (
    analyze_emotions_from_transcript,
    generate_emotion_summary
)
from config import SUPPORTED_LANGUAGES, SUPPORTED_AUDIO_FORMATS
# ... rest of the imports and code ... 