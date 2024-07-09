import torch
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1

print("Loading sentiment analysis pipeline...")
# sentiment_analyzer = pipeline("sentiment-analysis", model="daigo/bert-base-japanese-sentiment", device=device)
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=device)

def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result[0]['label'], result[0]['score']

# Test the function
text = "これはテストです。"
label, score = analyze_sentiment(text)
print(f"Sentiment: {label}, Score: {score}")
