# To verify everything is working

print("Sentiment analysis project is set up!")
print("Python is working correctly.")

# Test importing libraries
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("All libraries imported successfully!")

# Quick test with a sample sentence
test_sentence = "This chatbot helped me feel better today"
blob = TextBlob(test_sentence)
print(f"\nTest sentence: {test_sentence}")
print(f"TextBlob polarity: {blob.sentiment.polarity}")