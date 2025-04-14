# Download script for saving models locally
# save as download_models.py

import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Create directories for models
os.makedirs("models/distilgpt2", exist_ok=True)
os.makedirs("models/distilbert-sentiment", exist_ok=True)

print("Downloading and saving language model (distilgpt2)...")
# Download and save the language model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save locally
tokenizer.save_pretrained("models/distilgpt2")
model.save_pretrained("models/distilgpt2")
print("Language model saved successfully!")

print("Downloading and saving sentiment model...")
# Download and save sentiment analysis model
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
sentiment_analyzer.save_pretrained("models/distilbert-sentiment")
print("Sentiment model saved successfully!")

print("All models downloaded and saved locally!")