# Emotion Analysis 

## Introduction

This AI-powered system provides deep emotion analysis from text, helping you uncover hidden emotional patterns and insights with cutting-edge NLP techniques.

## Key Features

- Extracts key ideas from transcripts
- Analyzes emotions within text
- Generates comprehensive emotion reports
- Supports multiple languages

## Installation

### System Requirements

- Python 3.8+
- Required dependencies

### Install Dependencies

Run the following command to install the necessary libraries:

```bash
pip install -r requirements.txt
```

### Set Up Environment Variables

Create a `.env` file and add the following details:

```
OPENAI_API_KEY=your_openai_api_key
HUME_API_KEY=your_hume_api_key
```

## Usage

### 1. Analyze Emotions in Text

Use `analyzer.py` to analyze emotions in text:

```python
from analyzer import analyze_emotions_in_text
result = analyze_emotions_in_text("Today, I feel extremely happy!")
print(result)
```

### 2. Analyze Emotions from a Transcript

```python
from analyzer import analyze_emotions_from_transcript
test_transcript = "I’m so happy about the good news, but a little worried about the future."
result = analyze_emotions_from_transcript(test_transcript)
print(result)
```

### 3. Run the Web Application

The application uses `Streamlit`. Run it with the following command:

```bash
streamlit run app.py
```

## Reviewer’s Guide

### Verify Installation

1. Ensure all required dependencies are installed:

```bash
pip install -r requirements.txt
```

2. Check if API keys are properly set up:

```bash
echo $OPENAI_API_KEY
echo $HUME_API_KEY
```

### Testing the System

- Run the sample code in the **Usage** section to validate emotion analysis.
- Ensure JSON output is correctly formatted.
- For the web app, launch `localhost` in a browser after running `Streamlit`.



## Notes

- The system leverages OpenAI GPT for text analysis.
- API keys must be configured before use.

