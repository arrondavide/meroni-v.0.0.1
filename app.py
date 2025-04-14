# app.py - Flask server for Meroni with local model loading

from flask import Flask, render_template, request, jsonify
import os
import json
import re
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import random

app = Flask(__name__)

# Global model and tokenizer variables
model = None
tokenizer = None
sentiment_analyzer = None

# Load mental health response templates
try:
    with open('mental_health_responses.json', 'r') as f:
        response_templates = json.load(f)
except FileNotFoundError:
    # Fallback with basic templates if file isn't found
    response_templates = {
        "general": {
            "general": [
                "I'm here to listen. Would you like to tell me more about what's on your mind?",
                "Thank you for sharing that with me. How long have you been feeling this way?",
                "I appreciate you opening up. Is there something specific that would help you right now?"
            ]
        }
    }

# Function to load models from local storage
def load_models():
    global model, tokenizer, sentiment_analyzer
    
    print("Loading models from local storage...")
    
    # Check if models exist locally
    if os.path.exists("models/distilgpt2") and os.path.exists("models/distilbert-sentiment"):
        # Load language model from local storage
        print("Loading language model from local storage...")
        tokenizer = AutoTokenizer.from_pretrained("models/distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("models/distilgpt2")
        
        # Load sentiment analyzer from local storage
        print("Loading sentiment model from local storage...")
        sentiment_analyzer = pipeline('sentiment-analysis', model="models/distilbert-sentiment")
        
        print("All models loaded from local storage successfully!")
    else:
        print("Local models not found. Downloading from Hugging Face...")
        # Fall back to downloading from Hugging Face if local models don't exist
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
        print("Models downloaded from Hugging Face successfully!")

# Function to detect user emotional state
def detect_emotion(text):
    # Basic emotion detection using sentiment analysis
    sentiment_result = sentiment_analyzer(text)[0]
    
    # Additional keyword-based emotion detection
    emotions = {
        'anxious': r'\b(anxious|anxiety|worried|nervous|stress(ed)?|panic)\b',
        'sad': r'\b(sad|depressed|unhappy|miserable|down|blue|grief|upset)\b',
        'angry': r'\b(angry|annoyed|frustrated|irritated|mad|furious)\b',
        'happy': r'\b(happy|joy(ful)?|glad|pleased|delighted|content)\b',
        'fearful': r'\b(scared|afraid|fear(ful)?|terrified|phobia)\b',
        'lonely': r'\b(lonely|alone|isolated|abandoned|solitary)\b'
    }
    
    detected_emotions = []
    for emotion, pattern in emotions.items():
        if re.search(pattern, text.lower()):
            detected_emotions.append(emotion)
    
    # Determine primary emotion
    if detected_emotions:
        primary_emotion = detected_emotions[0]
    else:
        # Default to sentiment analysis result
        primary_emotion = 'positive' if sentiment_result['label'] == 'POSITIVE' else 'negative'
    
    return primary_emotion

# Function to detect topics being discussed
def detect_topic(text):
    topics = {
        'work': r'\b(work|job|career|boss|coworker|office|workplace)\b',
        'relationships': r'\b(relationship|partner|spouse|boyfriend|girlfriend|marriage|dating)\b',
        'family': r'\b(family|parent|mother|father|sibling|child|son|daughter)\b',
        'health': r'\b(health|sick|illness|disease|doctor|hospital|pain|symptoms)\b',
        'finance': r'\b(money|financial|debt|bills|afford|budget|cost|expense|salary|pay)\b',
        'self_esteem': r'\b(confidence|self[- ]worth|self[- ]esteem|self[- ]image|ugly|failure|worthless)\b',
        'future': r'\b(future|plan|goal|dream|aspiration|direction|purpose)\b',
        'stress': r'\b(overwhelm|burnout|stress|pressure|deadline|workload)\b'
    }
    
    detected_topics = []
    for topic, pattern in topics.items():
        if re.search(pattern, text.lower()):
            detected_topics.append(topic)
    
    return detected_topics if detected_topics else ['general']

# Function to select appropriate template based on emotion and topic
def select_response_template(emotion, topics):
    suitable_templates = []
    
    # First look for templates that match both emotion and topic
    for topic in topics:
        if emotion in response_templates and topic in response_templates[emotion]:
            suitable_templates.extend(response_templates[emotion][topic])
    
    # If no specific match, fall back to general templates for the emotion
    if not suitable_templates and emotion in response_templates and 'general' in response_templates[emotion]:
        suitable_templates = response_templates[emotion]['general']
    
    # If still no match, use completely general supportive responses
    if not suitable_templates and 'general' in response_templates and 'general' in response_templates['general']:
        suitable_templates = response_templates['general']['general']
    
    # Select a random template from suitable ones
    if suitable_templates:
        return random.choice(suitable_templates)
    else:
        # Fallback template if nothing else is found
        return "I'm here to listen and support you. Would you like to tell me more about how you're feeling?"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    global model, tokenizer, sentiment_analyzer
    
    # Load models if they haven't been loaded yet
    if model is None or tokenizer is None or sentiment_analyzer is None:
        load_models()
    
    data = request.json
    user_input = data.get('text', '')
    
    # Detect emotion and topic
    emotion = detect_emotion(user_input)
    topics = detect_topic(user_input)
    
    # Get appropriate response template
    template = select_response_template(emotion, topics)
    
    # Create a complete prompt with context
    prompt = f"User is feeling {emotion} and talking about {', '.join(topics)}.\nUser says: {user_input}\nMeroni (supportive mental health companion) responds using this guidance: {template}\nMeroni's response:"
    
    # Generate response
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=200,
        num_return_sequences=1,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )
    
    full_response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract just the assistant's response part
    try:
        response = full_response.split("Meroni's response:")[1].strip()
    except IndexError:
        # Fallback if splitting fails
        response = full_response.split("User says:")[1].strip()
    
    # Clean up response as needed
    response = re.sub(r'User:.*$', '', response, flags=re.DOTALL).strip()
    
    return jsonify({'response': response})

if __name__ == '__main__':
    # Preload models at startup
    load_models()
    app.run(debug=True, host='0.0.0.0')