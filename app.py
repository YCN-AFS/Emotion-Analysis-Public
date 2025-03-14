import streamlit as st
import json
from test import analyze_emotions_from_transcript, generate_emotion_summary
import plotly.graph_objects as go
import pandas as pd
import tempfile
import os
from openai import OpenAI
import io
import urllib.parse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def transcribe_audio(audio_file, source_language="en"):
    """
    Transcribes audio file using OpenAI's Whisper API.
    
    Args:
        audio_file: File object containing the audio data
        source_language: Source language code in ISO-639-1 format (default: 'en' for English)
        
    Returns:
        tuple: (original_text, translated_text) or (original_text, None) if no translation
    """
    try:
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as temp_file:
            temp_file.write(audio_file.getvalue())
            temp_file_path = temp_file.name
        
        # Open the temporary file for transcription
        with open(temp_file_path, "rb") as audio:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                language=source_language
            )
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        return transcript.text
        
    except Exception as e:
        st.error(f"Lỗi khi xử lý âm thanh: {str(e)}")
        return None

def create_emotion_radar_chart(emotions):
    """Creates a radar chart for top 10 emotion scores."""
    # Take top 10 emotions
    top_emotions = list(emotions.items())[:10]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[score for _, score in top_emotions],
        theta=[name for name, _ in top_emotions],
        fill='toself',
        name='Top Emotions'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat='.2f'
            )
        ),
        showlegend=False,
        title="Top 10 Most Prominent Emotions"
    )
    
    return fig

def create_emotion_bar_chart(emotions):
    """Creates a bar chart for top 10 emotion scores."""
    # Take top 10 emotions
    top_emotions = list(emotions.items())[:10]
    
    df = pd.DataFrame(top_emotions, columns=['name', 'score'])
    
    fig = go.Figure(go.Bar(
        x=df['score'],
        y=df['name'],
        orientation='h',
        text=[f"{score:.2f}" for score in df['score']],  # Format display only
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Top 10 Emotion Scores",
        xaxis_title="Score",
        yaxis_title="Emotion",
        xaxis=dict(
            range=[0, 1],
            tickformat='.2f'
        )
    )
    
    return fig

def convert_to_standard_format(analysis_results):
    """
    Converts the analysis results to the standard JSON format.
    
    Args:
        analysis_results (dict): Dictionary containing emotion analysis results
        
    Returns:
        list: List of dictionaries in the standard format
    """
    standardized_data = []
    
    for topic, emotions in analysis_results.items():
        emotion_list = [
            {
                "name": emotion_name,
                # Convert score from 0-1 to 0-100 scale
                "score": round(score * 100)
            }
            for emotion_name, score in emotions.items()
        ]
        
        standardized_data.append({
            "topic": topic,
            "emotions": emotion_list
        })
    
    return standardized_data

def normalize_emotion_scores(emotions):
    """
    Normalizes emotion scores so their sum equals 1.0
    
    Args:
        emotions (dict): Dictionary of emotion scores
        
    Returns:
        dict: Normalized emotion scores
    """
    # Get sum of all scores
    total = sum(emotions.values())
    
    # If total is 0 or 1, return original scores
    if total == 0 or total == 1:
        return emotions
    
    # Normalize scores
    normalized = {
        emotion: score/total if total > 0 else 0 
        for emotion, score in emotions.items()
    }
    
    return normalized

def analyze_emotions_in_text(text):
    """
    Analyzes emotions in a given text using LLM with 40 emotion categories.
    """
    prompt = f"""
    Analyze the emotions expressed in the following text. Rate each emotion on a scale of 0 to 1, where:
    - 0 means the emotion is not present at all
    - 1 means the emotion is very strongly present
    
    Remember that the sum of all scores should not exceed 1.0.
    
    Text: "{text}"
    
    Analyze for these 40 emotions:
    - joy
    - sadness
    - anger
    - fear
    - surprise
    - disgust
    - love
    - hate
    - anxiety
    - hope
    - disappointment
    - pride
    - shame
    - guilt
    - envy
    - confusion
    - curiosity
    - doubt
    - certainty
    - understanding
    - empathy
    - gratitude
    - admiration
    - contempt
    - respect
    - interest
    - boredom
    - determination
    - apathy
    - enthusiasm
    - stress
    - relief
    - satisfaction
    - frustration
    - overwhelm
    - nostalgia
    - amusement
    - serenity
    - melancholy
    - anticipation

    Return a JSON object with emotion names as keys and scores as values.
    """

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at detecting subtle emotional nuances in text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={ "type": "json_object" }
        )

        # Parse the JSON response
        emotion_scores = json.loads(completion.choices[0].message.content)
        
        # Normalize scores without rounding
        normalized_scores = normalize_emotion_scores(emotion_scores)
        
        # Convert to standard format
        emotions_list = [
            {
                "name": emotion,
                "score": score  # Remove rounding
            }
            for emotion, score in normalized_scores.items()
        ]
        
        # Sort by score in descending order
        emotions_list.sort(key=lambda x: x["score"], reverse=True)
        
        return emotions_list

    except Exception as e:
        print(f"Error analyzing emotions: {e}")
        return None

def extract_topics_and_sentences(transcript):
    """
    Extracts topics and their related sentences from the transcript.
    
    Args:
        transcript (str): The input text to analyze
        
    Returns:
        list: List of dictionaries containing topics and their related sentences
    """
    prompt = f"""
    Analyze the following text and identify distinct topics and their related sentences.
    For each topic:
    1. Give it a clear, concise name
    2. Extract the exact sentences from the text that relate to this topic
    
    Text:
    {transcript}
    
    Return a JSON array where each object has:
    - "topic": the topic name
    - "sentences": array of exact sentences from the text that discuss this topic
    
    Example format:
    {{
        "topics": [
            {{
                "topic": "Age goal",
                "sentences": ["So my goal, usually I want to be maybe a 90 or around 18 years old."]
            }}
        ]
    }}
    """

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at identifying topics and their context in text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={ "type": "json_object" }
        )

        # Parse the JSON response
        topics_data = json.loads(completion.choices[0].message.content)
        return topics_data.get('topics', [])

    except Exception as e:
        print(f"Error extracting topics: {e}")
        return []

def analyze_emotions_from_transcript(transcript):
    """
    Analyzes emotions from a transcript by topic and their related sentences.
    """
    # Get topics and their sentences
    topics_data = extract_topics_and_sentences(transcript)
    
    results = []
    
    # Analyze emotions for each topic
    for topic_data in topics_data:
        topic_name = topic_data['topic']
        sentences = topic_data['sentences']
        
        # Analyze emotions for each sentence
        sentence_emotions = []
        for sentence in sentences:
            emotions = analyze_emotions_in_text(sentence)
            if emotions:
                sentence_emotions.extend(emotions)

        # Calculate mean emotions for the topic
        if sentence_emotions:
            emotion_means = {}
            for emotion in sentence_emotions:
                name = emotion["name"]
                score = emotion["score"]
                if name in emotion_means:
                    emotion_means[name].append(score)
                else:
                    emotion_means[name] = [score]
            
            final_emotions = [
                {
                    "name": name,
                    "score": sum(scores) / len(scores)
                }
                for name, scores in emotion_means.items()
            ]
            
            # Sort by score in descending order
            final_emotions.sort(key=lambda x: x["score"], reverse=True)
            
            # Add to results with sentences
            results.append({
                "topic": topic_name,
                "sentences": sentences,  # Include related sentences
                "emotions": final_emotions
            })
        else:
            results.append({
                "topic": topic_name,
                "sentences": sentences,  # Include related sentences even if no emotions
                "emotions": []
            })

    return results

def main():
    # Initialize session state
    if 'original_text' not in st.session_state:
        st.session_state.original_text = None
    if 'translated_text' not in st.session_state:
        st.session_state.translated_text = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    st.set_page_config(
        page_title="Voice and Text Emotion Analysis",
        page_icon="😊",
        layout="wide"
    )
    
    st.title("😊 Voice and Text Emotion Analysis")
    st.markdown("""
    This application will analyze emotions from your text or voice by:
    1. Converting voice to text (if needed)
    2. Extracting main ideas
    3. Analyzing emotions for each idea
    4. Creating a comprehensive report
    """)
    
    # Input method selection
    input_method = st.radio(
        "Select input method:",
        ["Text", "Voice"]
    )
    
    text_input = None
    
    if input_method == "Text":
        st.header("Enter Text")
        text_input = st.text_area(
            "Enter text to analyze:",
            height=200,
            placeholder="Enter your text here..."
        )
        
        if text_input:
            st.code(text_input)
            st.caption("👆 Click the text block above and press Ctrl+C (Windows) or Cmd+C (Mac) to copy")
    else:
        st.header("Upload Audio File")
        
        source_language = st.selectbox(
            "Audio language:",
            options=["en", "vi", "ja", "ko", "zh", "fr", "de", "es", "it", "ru", "th", "id"],
            index=0,
            format_func=lambda x: {
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
            }.get(x, x)
        )
        
        st.markdown("""
        Supported formats:
        - MP3
        - MP4
        - M4A
        - WAV
        - WEBM
        """)
        
        audio_file = st.file_uploader(
            "Choose an audio file:",
            type=['mp3', 'mp4', 'm4a', 'wav', 'webm']
        )
        
        if audio_file:
            with st.spinner("Processing audio..."):
                text_input = transcribe_audio(audio_file, source_language)
                
                if text_input:
                    st.success("Conversion successful!")
                    
                    st.subheader("Transcribed Text:")
                    st.code(text_input)
                    st.caption("👆 Click the text block above and press Ctrl+C (Windows) or Cmd+C (Mac) to copy")
    
    if st.button("Analyze Emotions"):
        if text_input:
            with st.spinner("Analyzing... Please wait."):
                analysis_results = analyze_emotions_from_transcript(text_input)
                
                if analysis_results:
                    tab1, tab2, tab3 = st.tabs(["Detailed Results", "Charts", "Summary"])
                    
                    with tab1:
                        st.header("Detailed Analysis Results")
                        
                        with st.expander("View Full JSON", expanded=True):
                            full_json = json.dumps(analysis_results, indent=2, ensure_ascii=False)
                            st.code(full_json, language='json')
                            
                            st.caption("👆 Click the code block above and press Ctrl+C (Windows) or Cmd+C (Mac) to copy")
                        
                        for idx, result in enumerate(analysis_results):
                            with st.container():
                                st.subheader(f"Topic: {result['topic']}")
                                
                                with st.expander("Related Sentences", expanded=True):
                                    sentences_text = "\n".join(result['sentences'])
                                    st.code(sentences_text)
                                    st.caption("👆 Click the text block above and press Ctrl+C (Windows) or Cmd+C (Mac) to copy")
                                
                                with st.expander("Emotion Analysis", expanded=True):
                                    emotions_json = json.dumps(result['emotions'], indent=2, ensure_ascii=False)
                                    st.code(emotions_json, language='json')
                                    st.caption("👆 Click the code block above and press Ctrl+C (Windows) or Cmd+C (Mac) to copy")
                                
                                st.markdown("---")
                    
                    with tab2:
                        st.header("Analysis Charts")
                        for result in analysis_results:
                            st.subheader(f"Charts for: {result['topic']}")
                            
                            emotions_dict = {
                                emotion['name']: emotion['score']
                                for emotion in result['emotions']
                            }
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.plotly_chart(
                                    create_emotion_radar_chart(emotions_dict), 
                                    use_container_width=True
                                )
                            with col2:
                                st.plotly_chart(
                                    create_emotion_bar_chart(emotions_dict), 
                                    use_container_width=True
                                )
                    
                    with tab3:
                        st.header("Analysis Summary")
                        summary = generate_emotion_summary(analysis_results)
                        if summary:
                            st.markdown(summary)
                        else:
                            st.error("Could not generate summary.")
                else:
                    st.error("Analysis failed. Please try again.")
        else:
            st.warning("Please enter text or upload an audio file to analyze.")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Developed with ❤️ | Using OpenAI GPT-3.5 & Whisper</p>
    </div>
    """, unsafe_allow_html=True)

# Khởi tạo session state nếu chưa có
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def create_download_link(data, filename="data.json"):
    """
    Creates a download link for JSON data
    """
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    return f'<a href="data:application/json;charset=utf-8,{urllib.parse.quote(json_str)}" download="{filename}">📥 Tải xuống {filename}</a>'

if __name__ == "__main__":
    main() 