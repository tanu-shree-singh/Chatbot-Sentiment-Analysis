# Step 8.3: Sentiment Analysis on Full MHARD Dataset

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import time

print("=" * 60)
print("SENTIMENT ANALYSIS ON FULL MHARD DATASET")
print("=" * 60)

# Load the filtered chatbot data
print("\nLoading chatbot reviews...")
df = pd.read_csv('chatbot_reviews_full.csv')
print(f"Loaded {len(df):,} chatbot reviews")

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment scores (with error handling)
def get_vader_score(text):
    if pd.isna(text) or text == "":
        return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0}
    try:
        return vader_analyzer.polarity_scores(str(text))
    except:
        return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0}

def get_textblob_score(text):
    if pd.isna(text) or text == "":
        return {'polarity': 0, 'subjectivity': 0}
    try:
        blob = TextBlob(str(text))
        return {'polarity': blob.sentiment.polarity, 'subjectivity': blob.sentiment.subjectivity}
    except:
        return {'polarity': 0, 'subjectivity': 0}

# Apply sentiment analysis (this will take several minutes for 200k+ reviews)
print("\n🔄 Running sentiment analysis on all reviews...")
print("   (This may take 5-10 minutes for the full dataset)")

start_time = time.time()

vader_scores = df['review'].apply(get_vader_score)
textblob_scores = df['review'].apply(get_textblob_score)

# Extract scores
df['vader_compound'] = vader_scores.apply(lambda x: x['compound'])
df['vader_positive'] = vader_scores.apply(lambda x: x['pos'])
df['vader_negative'] = vader_scores.apply(lambda x: x['neg'])
df['vader_neutral'] = vader_scores.apply(lambda x: x['neu'])
df['textblob_polarity'] = textblob_scores.apply(lambda x: x['polarity'])
df['textblob_subjectivity'] = textblob_scores.apply(lambda x: x['subjectivity'])

end_time = time.time()
print(f"✅ Analysis complete in {end_time - start_time:.2f} seconds")

# Classify sentiment
def classify_sentiment(compound):
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['vader_compound'].apply(classify_sentiment)

# ============================================
# RESULTS SUMMARY
# ============================================

print("\n" + "=" * 60)
print("FULL DATASET SENTIMENT RESULTS")
print("=" * 60)

print(f"\n📊 Total chatbot reviews analyzed: {len(df):,}")
print(f"\n📊 Average VADER compound score: {df['vader_compound'].mean():.3f}")
print(f"📊 Average TextBlob polarity: {df['textblob_polarity'].mean():.3f}")

print(f"\n📊 Sentiment Distribution:")
sentiment_counts = df['sentiment'].value_counts()
for sentiment, count in sentiment_counts.items():
    print(f"   {sentiment}: {count:,} ({count/len(df)*100:.1f}%)")

print(f"\n📊 Sentiment by Rating:")
sentiment_by_rating = pd.crosstab(df['rating'], df['sentiment'])
print(sentiment_by_rating)

# Calculate correlation between rating and sentiment
correlation = df['rating'].corr(df['vader_compound'])
print(f"\n📊 Correlation between user rating and VADER sentiment: {correlation:.3f}")

# ============================================
# PER-APP ANALYSIS
# ============================================

print("\n" + "=" * 60)
print("PER-APP SENTIMENT ANALYSIS")
print("=" * 60)

app_summary = df.groupby('app_name').agg({
    'vader_compound': 'mean',
    'rating': 'mean',
    'review': 'count'
}).rename(columns={'review': 'count'}).sort_values('count', ascending=False)

print(app_summary)

# Save results
df.to_csv('sentiment_results_full.csv', index=False)
print("\n✅ Full results saved to 'sentiment_results_full.csv'")
print("✅ App summary saved above")