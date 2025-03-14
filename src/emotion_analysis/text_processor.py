import re
from .utils import get_openai_client

# Get OpenAI client instance
openai_client = get_openai_client()

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
        ideas = [re.sub(r"^\d+\.\s*", "", line).strip() 
                for line in idea_string.splitlines() 
                if line.strip()]
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

        sentences = [sentence.strip() 
                    for sentence in completion.choices[0].message.content.split(". ") 
                    if sentence.strip()]
        return sentences

    except Exception as e:
        print(f"Error calling OpenAI API to find sentences: {e}")
        return None 