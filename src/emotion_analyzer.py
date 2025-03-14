import os
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from hume import HumeClient
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class EmotionAnalyzer:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('HUME_API_KEY')
        if not self.api_key:
            raise ValueError("HUME_API_KEY not found in environment variables")
        
        self.client = HumeClient(api_key=self.api_key)
        self.data = None
        
    def debug_json_content(self, content):
        """Debug helper to identify JSON parsing issues"""
        try:
            # Try parsing small chunks to identify where the problem is
            chunk_size = 1000
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                try:
                    json.loads("{" + chunk + "}")
                except json.JSONDecodeError as e:
                    print(f"\nPotential issue at position {i + e.pos}:")
                    print(content[i + e.pos - 50:i + e.pos + 50])
                    print("^--- Error here")
            return False
        except Exception as e:
            print(f"Debug error: {str(e)}")
            return False

    async def load_data(self, file_path):
        """Load emotion data from the RTF file containing JSON data"""
        try:
            with open(file_path, 'rb') as f:  # Open in binary mode
                content = f.read().decode('utf-8', errors='ignore')
                
                # Find the actual JSON content
                # Look for common JSON start patterns
                possible_starts = [
                    '{"id":', '{\n"id":', '{ "id":', 
                    '{"title":', '{\n"title":', '{ "title":'
                ]
                
                json_start = -1
                for start in possible_starts:
                    pos = content.find(start)
                    if pos != -1:
                        json_start = pos
                        break
                
                if json_start == -1:
                    print("Could not find JSON start marker")
                    return False
                
                # Find the last closing brace
                json_end = content.rfind('}')
                if json_end == -1:
                    print("Could not find JSON end marker")
                    return False
                
                # Extract the JSON content
                json_content = content[json_start:json_end + 1]
                
                # Clean up RTF artifacts and fix JSON structure
                import re
                
                # First pass: Clean up basic RTF control sequences
                json_content = re.sub(r'\\[a-zA-Z0-9]+(?=[\s,}])', '', json_content)
                
                # Second pass: Fix array structures and escaped characters
                json_content = re.sub(r'\\\{', '{', json_content)  # Fix escaped opening braces
                json_content = re.sub(r'\\\}', '}', json_content)  # Fix escaped closing braces
                json_content = re.sub(r'\\(?!["\\/bfnrt])', '', json_content)  # Remove unnecessary escapes but keep valid JSON escapes
                
                # Third pass: Fix specific JSON structure issues
                json_content = re.sub(r'time":}', 'time":', json_content)  # Fix malformed time objects
                json_content = re.sub(r'\[}(?=")', '[{', json_content)  # Fix array starts
                json_content = re.sub(r'(?<=}),(?=})', '', json_content)  # Remove invalid commas
                
                # Fourth pass: Handle multiple JSON objects
                # Split content on potential object boundaries and clean up
                objects = re.split(r'},\s*{', json_content)
                if len(objects) > 1:
                    # Reconstruct as a single object with merged properties
                    merged_obj = {}
                    for obj in objects:
                        # Add missing braces if needed
                        if not obj.startswith('{'): obj = '{' + obj
                        if not obj.endswith('}'): obj = obj + '}'
                        
                        try:
                            obj_data = json.loads(obj)
                            # Merge arrays and objects
                            for key, value in obj_data.items():
                                if key in merged_obj:
                                    if isinstance(value, list) and isinstance(merged_obj[key], list):
                                        merged_obj[key].extend(value)
                                    elif isinstance(value, dict) and isinstance(merged_obj[key], dict):
                                        merged_obj[key].update(value)
                                    else:
                                        # Keep the latest non-list/non-dict value
                                        merged_obj[key] = value
                                else:
                                    merged_obj[key] = value
                        except json.JSONDecodeError:
                            continue
                    
                    # Convert back to JSON string
                    json_content = json.dumps(merged_obj)
                
                # Clean up whitespace while preserving structure
                json_content = re.sub(r'\s+', ' ', json_content)
                json_content = json_content.strip()
                
                # Save intermediate state for debugging
                with open('debug_json.json', 'w', encoding='utf-8') as f:
                    f.write(json_content)
                
                # Try to parse the JSON
                try:
                    self.data = json.loads(json_content)
                    print("Successfully loaded JSON data")
                    
                    # Debug: Print data structure
                    print("\nData structure overview:")
                    print(f"Keys in data: {list(self.data.keys())}")
                    
                    # Print root-level emotion data if present
                    if 'name' in self.data and 'score' in self.data:
                        print(f"Root-level emotion: {self.data['name']} (score: {self.data['score']})")
                    
                    if 'transcript' in self.data:
                        print(f"Number of transcript entries: {len(self.data['transcript'])}")
                        if self.data['transcript']:
                            print("First transcript entry keys:", list(self.data['transcript'][0].keys()))
                            if 'predictions' in self.data['transcript'][0]:
                                print("First prediction structure:", self.data['transcript'][0]['predictions'])
                    
                    if 'topics' in self.data:
                        topics = self.data.get('topics', [])
                        print(f"Number of topics: {len(topics)}")
                        if topics:
                            print("First topic structure:", topics[0])
                    
                    # Validate required data structure
                    has_emotion_data = (
                        ('name' in self.data and 'score' in self.data) or
                        any(pred.get('emotions') for entry in self.data.get('transcript', [])
                            for pred in entry.get('predictions', []))
                    )
                    
                    if not has_emotion_data:
                        print("\nWarning: No emotion data found in the file")
                    
                    return True
                except json.JSONDecodeError as je:
                    print(f"JSON decode error: {str(je)}")
                    print("\nAttempting to debug JSON content...")
                    print(f"Error location: character {je.pos}")
                    context = json_content[max(0, je.pos-100):min(len(json_content), je.pos+100)]
                    print(context)
                    print(" " * (min(100, je.pos)) + "^")
                    
                    # Additional debug info
                    print("\nCharacters around error position:")
                    error_chars = json_content[max(0, je.pos-5):min(len(json_content), je.pos+5)]
                    print(f"ASCII values: {[ord(c) for c in error_chars]}")
                    return False
                
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
        
    def analyze_transcript_emotions(self, data):
        """Analyze emotions from transcript data"""
        emotion_scores = {}
        timestamps = []
        
        # Check if we have root-level emotion data
        if 'name' in data and 'score' in data:
            name = data['name']
            score = data['score']
            emotion_scores[name] = [score]
            if 'startTime' in data:
                timestamps.append(data.get('startTime', 0))
            elif 'start_words' in data:
                timestamps.append(0)  # Use 0 as default start time
        
        # Also check transcript entries
        transcript = data.get('transcript', [])
        for entry in transcript:
            text = entry.get('text', '')
            start_time = entry.get('startTimestamp', 0)
            predictions = entry.get('predictions', [])
            
            for prediction in predictions:
                for emotion in prediction.get('emotions', []):
                    name = emotion['name']
                    score = emotion['score']
                    if name not in emotion_scores:
                        emotion_scores[name] = []
                    emotion_scores[name].append(score)
                    if len(timestamps) < len(emotion_scores[name]):
                        timestamps.append(start_time)
        
        return emotion_scores, timestamps
    
    def analyze_topic_emotions(self, data):
        """Analyze emotions by topic"""
        topic_emotions = {}
        
        # Handle root-level topic data
        if all(key in data for key in ['topic', 'name', 'score']):
            topic_name = data.get('topic', '').strip()
            emotion_name = data['name']
            emotion_score = data['score']
            if topic_name:  # Only add if we have a non-empty topic name
                topic_emotions[topic_name] = {emotion_name: emotion_score}
        
        # Also check topics array
        topics = data.get('topics', [])
        for topic in topics:
            topic_name = topic.get('topic', '').strip()
            emotions = topic.get('emotions', {})
            if topic_name:
                if topic_name in topic_emotions:
                    topic_emotions[topic_name].update(emotions)
                else:
                    topic_emotions[topic_name] = emotions
        
        # If no topics found but we have root emotion data, create a default topic
        if not topic_emotions and 'name' in data and 'score' in data:
            # Try to extract a meaningful topic from the text if available
            text = next((entry.get('text', '') for entry in data.get('transcript', []) if entry.get('text')), '')
            if text:
                # Clean up the text and use it as the topic
                topic_name = text.strip()
                # Remove any trailing punctuation
                topic_name = topic_name.rstrip('.!?')
                # Limit length if needed
                if len(topic_name) > 50:
                    topic_name = topic_name[:47] + "..."
            else:
                topic_name = "General Analysis"
            
            topic_emotions[topic_name] = {data['name']: data['score']}
        
        return topic_emotions
        
    def generate_insights(self, data):
        """Generate insights from the emotion analysis data"""
        # Get title, clean it up and provide a default if empty
        title = data.get('title', '').strip()
        if not title:
            # Try to use the text from the first topic
            if data.get('topic'):
                title = data['topic'].strip()
            
            # If still no title, try to generate from transcript
            if not title:
                transcript = data.get('transcript', [])
                if transcript and transcript[0].get('text'):
                    text = transcript[0]['text'].strip()
                    # Use first sentence if available, otherwise use first 50 chars
                    title = text.split('.')[0].strip() if '.' in text else text[:50].strip()
                    # Remove any trailing punctuation
                    title = title.rstrip('.,!?')
                
                # If still no title, use a default
                if not title:
                    title = "the conversation"
        
        # Clean up title if it's a statement
        title = title.rstrip('.,!?')
        if title.lower().endswith(' here'):
            title = title[:-5].rstrip()
        if title.lower().startswith('nothing is being '):
            title = title[15:].strip()
        if not title:
            title = "the conversation"
        
        insights = {
            'metadata': {
                'title': title,
                'created_at': data.get('createdAt', ''),
                'decision': data.get('decision', '')
            },
            'transcript_analysis': {},
            'topic_analysis': {},
            'overall_summary': '',
            'key_insight': '',  # New field for actionable insight
            'emotional_context': '',  # New field for contextual understanding
            'suggested_action': ''  # New field for action recommendation
        }
        
        # Analyze emotions
        emotion_scores, timestamps = self.analyze_transcript_emotions(data)
        
        # Calculate overall emotion trends
        avg_scores = {}
        for name, scores in emotion_scores.items():
            if scores:  # Only calculate if we have scores
                avg_scores[name] = np.mean(scores)
        
        # Sort emotions by average score
        sorted_emotions = sorted(
            avg_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Analyze topics
        topic_emotions = self.analyze_topic_emotions(data)
        
        # Generate insights
        insights['transcript_analysis'] = {
            'dominant_emotions': sorted_emotions[:3],
            'emotion_trends': {
                name: list(zip(timestamps, scores))
                for name, scores in emotion_scores.items()
            },
            'average_scores': avg_scores
        }
        
        # Calculate topic scores
        topic_scores = []
        for topic, emotions in topic_emotions.items():
            if emotions:  # Only include topics with emotion data
                score = sum(emotions.values())
                topic_scores.append((topic, score))
        
        topic_scores.sort(key=lambda x: x[1], reverse=True)
        
        insights['topic_analysis'] = {
            'topics': topic_emotions,
            'top_emotional_topics': topic_scores[:3]
        }
        
        # Generate overall summary
        top_emotions = [f"{e[0]} ({e[1]:.2f})" for e in sorted_emotions[:3]]
        
        # Format topic scores differently if we only have one topic with one emotion
        if len(topic_scores) == 1 and len(topic_emotions[topic_scores[0][0]]) == 1:
            # For single topic with single emotion, don't repeat the score
            topic = topic_scores[0][0]
            summary_parts = []
            if top_emotions:
                summary_parts.append(f"the primary emotion detected was {top_emotions[0]}")
            if topic:
                # Check if the topic is the same as the title to avoid repetition
                if topic.lower() != title.lower():
                    # Clean up topic if it's a statement
                    topic = topic.rstrip('.,!?')
                    if topic.lower().endswith(' here'):
                        topic = topic[:-5].rstrip()
                    summary_parts.append(f"in the context of {topic}")
        else:
            # For multiple topics or emotions, show scores
            top_topics = [f"{t[0]} ({t[1]:.2f})" for t in topic_scores[:3]]
            summary_parts = []
            if top_emotions:
                if len(top_emotions) == 1:
                    summary_parts.append(f"the primary emotion detected was {top_emotions[0]}")
                else:
                    summary_parts.append(f"the main emotions detected were {', '.join(top_emotions)}")
            if top_topics:
                if len(top_topics) == 1:
                    summary_parts.append(f"with the highest emotional intensity during {top_topics[0]}")
                else:
                    summary_parts.append(f"with notable emotional intensity during {', '.join(top_topics)}")
        
        if summary_parts:
            insights['overall_summary'] = f"In {title}, {' and '.join(summary_parts)}."
        else:
            insights['overall_summary'] = f"No significant emotional patterns were detected in {title}."
            
        # Generate key insight based on emotional patterns
        if sorted_emotions:
            primary_emotion = sorted_emotions[0]
            emotion_name = primary_emotion[0]
            emotion_score = primary_emotion[1]
            
            # Map emotions to insights and actions
            emotion_insights = {
                'Tiredness': {
                    'insight': 'High levels of fatigue detected in the voice',
                    'context': 'Fatigue can impact decision-making and communication clarity',
                    'action': 'Consider scheduling important discussions during higher energy periods'
                },
                'Confidence': {
                    'insight': 'Strong confidence detected in the voice',
                    'context': 'Confidence often correlates with clear communication and leadership',
                    'action': 'Leverage this confidence to drive important initiatives or discussions'
                },
                'Uncertainty': {
                    'insight': 'Notable uncertainty detected in the voice',
                    'context': 'Uncertainty might indicate areas needing clarification or support',
                    'action': 'Consider following up with specific questions to address unclear points'
                }
                # Add more emotion mappings as needed
            }
            
            # Get insight mapping for the primary emotion
            emotion_mapping = emotion_insights.get(emotion_name, {
                'insight': f'Significant {emotion_name.lower()} detected in the voice',
                'context': f'The level of {emotion_name.lower()} may influence the communication dynamics',
                'action': 'Consider how this emotional state might affect the conversation objectives'
            })
            
            insights['key_insight'] = f"{emotion_mapping['insight']} (intensity: {emotion_score:.2f})"
            insights['emotional_context'] = emotion_mapping['context']
            insights['suggested_action'] = emotion_mapping['action']
        
        return insights
        
    def visualize_emotions(self, data, output_path):
        """Create visualizations of emotion analysis results"""
        # Create directory for visualizations
        output_dir = Path(output_path).parent
        output_dir.mkdir(exist_ok=True)
        
        # 1. Overall emotion distribution
        transcript = data.get('transcript', [])
        emotion_scores, timestamps = self.analyze_transcript_emotions(data)
        
        plt.figure(figsize=(12, 6))
        avg_scores = {name: np.mean(scores) for name, scores in emotion_scores.items()}
        emotions = list(avg_scores.keys())
        scores = list(avg_scores.values())
        
        sns.barplot(x=emotions, y=scores)
        plt.xticks(rotation=45, ha='right')
        plt.title('Overall Emotion Distribution')
        plt.xlabel('Emotion')
        plt.ylabel('Average Score')
        plt.tight_layout()
        plt.savefig(f"{output_path}_overall_emotions.png")
        plt.close()
        
        # 2. Emotion trends over time
        plt.figure(figsize=(15, 8))
        for emotion, scores in emotion_scores.items():
            if len(scores) > 0:  # Only plot if we have data
                times = [(t - timestamps[0])/1000 for t in timestamps[:len(scores)]]  # Convert to seconds
                plt.plot(times, scores, label=emotion, alpha=0.7)
        
        plt.title('Emotion Trends Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Emotion Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{output_path}_emotion_trends.png")
        plt.close()
        
        # 3. Topic-based emotion analysis
        topics = data.get('topics', [])
        topic_emotions = self.analyze_topic_emotions(data)
        
        if topic_emotions:
            plt.figure(figsize=(12, 6))
            topic_scores = [(topic, sum(emotions.values())) 
                          for topic, emotions in topic_emotions.items()]
            topic_scores.sort(key=lambda x: x[1], reverse=True)
            
            topics, scores = zip(*topic_scores)
            sns.barplot(x=list(topics), y=list(scores))
            plt.xticks(rotation=45, ha='right')
            plt.title('Emotional Intensity by Topic')
            plt.xlabel('Topic')
            plt.ylabel('Total Emotional Score')
            plt.tight_layout()
            plt.savefig(f"{output_path}_topic_emotions.png")
            plt.close()

async def main():
    analyzer = EmotionAnalyzer()
    
    # Load the emotion data
    data_file = Path("Emotions_data.rtf")
    if not data_file.exists():
        print(f"Error: {data_file} not found")
        return
        
    if not await analyzer.load_data(data_file):
        print("Failed to load emotion data")
        return
    
    # Generate insights
    insights = analyzer.generate_insights(analyzer.data)
    
    # Save insights to JSON
    with open("emotion_analysis_results.json", "w", encoding='utf-8') as f:
        json.dump(insights, f, indent=2, ensure_ascii=False)
    
    # Create visualizations
    analyzer.visualize_emotions(analyzer.data, "emotion_analysis_results")
    
    print("\nAnalysis complete! Results saved to:")
    print("- emotion_analysis_results.json")
    print("- emotion_analysis_results_overall_emotions.png")
    print("- emotion_analysis_results_emotion_trends.png")
    print("- emotion_analysis_results_topic_emotions.png")
    
    # Print summary
    print("\nSummary:")
    print(insights['overall_summary'])
    
if __name__ == "__main__":
    asyncio.run(main()) 