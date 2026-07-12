# Step 8: Load and Filter Full MHARD Dataset

import pandas as pd
import numpy as np

print("=" * 60)
print("LOADING FULL MHARD DATASET")
print("=" * 60)

# Load the full dataset (this may take 1-2 minutes)
print("\nLoading MHARD_dataset.csv... This may take a moment.")
df = pd.read_csv('MHARD_dataset.csv')

print(f"\n✅ Dataset loaded successfully!")
print(f"Total reviews: {len(df):,}")
print(f"Columns: {list(df.columns)}")

# Check for missing values
print("\n📊 Missing Values:")
print(df.isnull().sum())

# Display basic statistics
print("\n📊 Rating Distribution (Ground Truth):")
print(df['rating'].value_counts().sort_index())

# ============================================
# FILTER FOR CHATBOT APPS ONLY
# ============================================

# List of conversational chatbot apps from the dataset [citation:1]
chatbot_apps = [
    'woebot', 'wysa', 'youper', 'sevencups', 'talkspace', 
    'replika', 'minddoc', 'cerebral', 'teencounseling', 'regain'
]

# Filter the dataset
df_chatbots = df[df['app_name'].isin(chatbot_apps)]

print("\n" + "=" * 60)
print("CHATBOT-ONLY ANALYSIS")
print("=" * 60)
print(f"\nTotal chatbot reviews: {len(df_chatbots):,}")
print(f"\nReviews per chatbot app:")
print(df_chatbots['app_name'].value_counts())

# Save filtered dataset for faster future loading
df_chatbots.to_csv('chatbot_reviews_full.csv', index=False)
print("\n✅ Filtered chatbot data saved to 'chatbot_reviews_full.csv'")

# Display sample of chatbot reviews
print("\n📝 Sample of chatbot reviews:")
print(df_chatbots[['app_name', 'rating', 'review']].head(10))