import json
import numpy as np
from openai import OpenAI
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def get_ideas_from_transcript(transcript):
    """
    Uses an LLM to extract a list of ideas from the transcript.

    Args:
        transcript (str): The transcript of the voice memo.

    Returns:
        list: A list of strings, where each string is an idea.
    """
    prompt = f"""
    Please extract a list of the main ideas discussed in the following transcript:

    {transcript}

    Return the ideas as a numbered list, with each idea on a new line. Be concise.
    """

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        idea_string = completion.choices[0].message.content
        ideas = [re.sub(r"^\d+\.\s*", "", line).strip() for line in idea_string.splitlines() if line.strip()]
        return ideas

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

def find_sentences_for_idea(idea, transcript):
    """
    Finds sentences in the transcript that are relevant to a given idea.

    Args:
        idea (str): The idea to search for.
        transcript (str): The transcript of the voice memo.

    Returns:
        list: A list of strings, where each string is a sentence relevant to the idea.
    """
    prompt = f"""
    Given the following idea: '{idea}'

    Find sentences in the transcript that are relevant to this idea. Return only the sentences. If no sentences are relevant, return an empty list.

    Transcript:
    {transcript}
    """

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        sentences = [sentence.strip() for sentence in completion.choices[0].message.content.split(". ") if sentence.strip()]
        return sentences

    except Exception as e:
        print(f"Error calling OpenAI API to find sentences: {e}")
        return None

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

        # Parse the JSON response
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
    # 1. Get ideas from the transcript using LLM
    ideas = get_ideas_from_transcript(transcript)
    if not ideas:
        print("Failed to extract ideas from transcript. Aborting.")
        return None

    results = {}

    # 2. Iterate through the ideas
    for idea in ideas:
        # 3. Find sentences for the idea
        sentences = find_sentences_for_idea(idea, transcript)

        if not sentences:
            print(f"No sentences found for idea: {idea}")
            results[idea] = {}
            continue

        # 4. Analyze emotions for each sentence
        sentence_emotions = []
        for sentence in sentences:
            emotions = analyze_emotions_in_text(sentence)
            if emotions:
                sentence_emotions.append(emotions)

        # 5. Calculate the mean emotions for this idea
        if sentence_emotions:
            # Get all emotion names
            all_emotion_names = set()
            for emotion_dict in sentence_emotions:
                all_emotion_names.update(emotion_dict.keys())

            # Calculate mean for each emotion
            emotion_sums = {emotion: 0 for emotion in all_emotion_names}
            for emotion_dict in sentence_emotions:
                for emotion, value in emotion_dict.items():
                    emotion_sums[emotion] += value

            mean_emotions = {emotion: emotion_sums[emotion] / len(sentence_emotions) 
                          for emotion in all_emotion_names}
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

def main():
    transcript_file_path = "transcript.txt"

    try:
        with open(transcript_file_path, "r", encoding="utf-8") as f:
            transcript = f.read()

        # Analyze emotions from transcript
        analysis_results = analyze_emotions_from_transcript(transcript)

        if analysis_results:
            print("\nEmotion Analysis Results:")
            print(json.dumps(analysis_results, indent=2))
            
            # Generate comprehensive summary
            summary = generate_emotion_summary(analysis_results)
            if summary:
                print("\nEmotional Analysis Summary:")
                print(summary)
        else:
            print("Emotion analysis failed.")

    except FileNotFoundError:
        print(f"Error: Could not find transcript file at {transcript_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()