import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_openai_client():
    """
    Creates and returns an OpenAI client instance using API key from environment variables.
    
    Returns:
        OpenAI: An initialized OpenAI client instance
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    return OpenAI(api_key=api_key) 