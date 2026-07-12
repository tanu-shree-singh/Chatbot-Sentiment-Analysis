# Quick script to recreate the filtered chatbot dataset
import pandas as pd

print("Loading full MHARD dataset...")
df = pd.read_csv('MHARD_dataset.csv')

chatbot_apps = [
    'woebot', 'wysa', 'youper', 'sevencups', 'talkspace', 
    'replika', 'minddoc', 'cerebral', 'teencounseling', 'regain'
]

df_chatbots = df[df['app_name'].isin(chatbot_apps)]
print(f"Found {len(df_chatbots):,} chatbot reviews")

df_chatbots.to_csv('chatbot_reviews_full.csv', index=False)
print("✅ Saved to 'chatbot_reviews_full.csv'")