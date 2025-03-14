import json
from openai import OpenAI
from .text_processor import get_ideas_from_transcript, find_sentences_for_idea
from .utils import get_openai_client

# Get OpenAI client instance
openai_client = get_openai_client()

def analyze_emotions_in_text(text):
    """
    Analyzes emotions in a given text using LLM.

    Args:
        text (str): The text to analyze for emotions.

    Returns:
        dict: A dictionary containing emotion scores.
    """
    prompt = f"""
    Analyze the emotions expressed in the following text. Rate each emotion on a scale of 0 to 1, where:
    - 0 means the emotion is not present at all
    - 1 means the emotion is very strongly present

    Text: "{text}"

    Return a JSON object with emotion scores for the following emotions:
    - joy
    - sadness
    - anger
    - fear
    - surprise
    - disgust
    - trust
    - anticipation
    - love
    - anxiety
    - confusion
    - doubt
    - interest
    - pain

    Format the response as a valid JSON object with emotion names as keys and scores as values.
    """

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        emotion_scores = json.loads(completion.choices[0].message.content)
        return emotion_scores

    except Exception as e:
        print(f"Error analyzing emotions: {e}")
        return None

def analyze_emotions_from_transcript(transcript):
    """
    Analyzes emotions from a transcript.

    Args:
        transcript (str): The transcript of the voice memo.

    Returns:
        dict: A dictionary where keys are ideas and values are dictionaries of emotion scores.
    """
    ideas = get_ideas_from_transcript(transcript)
    if not ideas:
        print("Failed to extract ideas from transcript. Aborting.")
        return None

    results = {}

    for idea in ideas:
        sentences = find_sentences_for_idea(idea, transcript)

        if not sentences:
            print(f"No sentences found for idea: {idea}")
            results[idea] = {}
            continue

        sentence_emotions = []
        for sentence in sentences:
            emotions = analyze_emotions_in_text(sentence)
            if emotions:
                sentence_emotions.append(emotions)

        if sentence_emotions:
            all_emotion_names = set()
            for emotion_dict in sentence_emotions:
                all_emotion_names.update(emotion_dict.keys())

            emotion_sums = {emotion: 0 for emotion in all_emotion_names}
            for emotion_dict in sentence_emotions:
                for emotion, value in emotion_dict.items():
                    emotion_sums[emotion] += value

            mean_emotions = {
                emotion: emotion_sums[emotion] / len(sentence_emotions) 
                for emotion in all_emotion_names
            }
            results[idea] = mean_emotions
        else:
            print(f"No emotion data found for sentences related to idea: {idea}")
            results[idea] = {}

    return results

def generate_emotion_summary(analysis_results):
    """
    Generates a comprehensive summary of emotions based on the analysis results.

    Args:
        analysis_results (dict): A dictionary where keys are ideas and values are
                               dictionaries of emotion scores.

    Returns:
        str: A detailed summary of the emotional analysis.
    """
    prompt = f"""
    Based on the following emotion analysis results, provide a comprehensive summary of the speaker's emotional state and attitudes:

    {json.dumps(analysis_results, indent=2)}

    Please include:
    1. Overall emotional tone
    2. Key emotional patterns across different ideas
    3. Notable emotional shifts or contrasts
    4. Potential underlying attitudes or concerns
    5. Recommendations for addressing any emotional concerns

    Format the response in a clear, structured manner.
    """

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content

    except Exception as e:
        print(f"Error generating emotion summary: {e}")
        return None 