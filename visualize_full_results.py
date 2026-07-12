# Step 9: Publication-Ready Visualizations for Full Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set professional style for publication
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Load the results
df = pd.read_csv('sentiment_results_full.csv')

print("=" * 60)
print("CREATING PUBLICATION FIGURES")
print("=" * 60)

# ============================================
# FIGURE 1: Sentiment Distribution (Pie Chart)
# ============================================
fig1, ax1 = plt.subplots(figsize=(8, 8))

sentiment_counts = df['sentiment'].value_counts()
colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # Green, Red, Gray
explode = (0.05, 0.05, 0)  # Slightly explode positive and negative

ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90, explode=explode, textprops={'fontsize': 14})
ax1.set_title('Sentiment Distribution of Mental Health Chatbot Reviews\n(N = 28,487)', 
              fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('figure1_sentiment_pie.png', dpi=300, bbox_inches='tight')
print("✅ Figure 1 saved: 'figure1_sentiment_pie.png'")

# ============================================
# FIGURE 2: Sentiment Score by Rating (Box Plot)
# ============================================
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Create box plot
sns.boxplot(x='rating', y='vader_compound', data=df, ax=ax2, palette='viridis')
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Neutral (0)')
ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Moderate Positive')
ax2.set_xlabel('User Star Rating (1-5)', fontsize=12)
ax2.set_ylabel('VADER Sentiment Score (-1 = Negative, +1 = Positive)', fontsize=12)
ax2.set_title('Sentiment Score vs. User Rating\nCorrelation: r = 0.612', fontsize=14, fontweight='bold')
ax2.legend()
ax2.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig('figure2_sentiment_by_rating.png', dpi=300, bbox_inches='tight')
print("✅ Figure 2 saved: 'figure2_sentiment_by_rating.png'")

# ============================================
# FIGURE 3: Per-App Sentiment Comparison (Bar Chart)
# ============================================
fig3, ax3 = plt.subplots(figsize=(12, 6))

# Calculate per-app average sentiment
app_sentiment = df.groupby('app_name')['vader_compound'].mean().sort_values(ascending=True)
app_colors = ['#e74c3c' if x < 0.3 else '#f39c12' if x < 0.5 else '#2ecc71' for x in app_sentiment.values]

bars = ax3.barh(range(len(app_sentiment)), app_sentiment.values, color=app_colors)
ax3.set_yticks(range(len(app_sentiment)))
ax3.set_yticklabels(app_sentiment.index, fontsize=10)
ax3.axvline(x=0, color='black', linewidth=0.5)
ax3.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Moderate Positive')
ax3.set_xlabel('Average VADER Sentiment Score (-1 to +1)', fontsize=12)
ax3.set_title('Sentiment Score by Chatbot Platform\n(N = 28,487 reviews)', fontsize=14, fontweight='bold')
ax3.set_xlim(-0.2, 0.8)
ax3.legend()

# Add value labels on bars
for i, v in enumerate(app_sentiment.values):
    ax3.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('figure3_app_sentiment_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Figure 3 saved: 'figure3_app_sentiment_comparison.png'")

# ============================================
# FIGURE 4: Sentiment Distribution by Rating (Stacked Bar)
# ============================================
fig4, ax4 = plt.subplots(figsize=(10, 6))

# Create cross-tabulation
sentiment_by_rating = pd.crosstab(df['rating'], df['sentiment'])
sentiment_by_rating_percent = sentiment_by_rating.div(sentiment_by_rating.sum(axis=1), axis=0) * 100

sentiment_by_rating_percent.plot(kind='bar', stacked=True, ax=ax4, 
                                  color=['#2ecc71', '#95a5a6', '#e74c3c'])
ax4.set_xlabel('User Star Rating (1-5)', fontsize=12)
ax4.set_ylabel('Percentage of Reviews (%)', fontsize=12)
ax4.set_title('Sentiment Composition by Star Rating', fontsize=14, fontweight='bold')
ax4.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
ax4.set_ylim(0, 100)
ax4.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('figure4_sentiment_by_rating_stacked.png', dpi=300, bbox_inches='tight')
print("✅ Figure 4 saved: 'figure4_sentiment_by_rating_stacked.png'")

# ============================================
# FIGURE 5: Top vs Bottom Performing Apps
# ============================================
fig5, ax5 = plt.subplots(figsize=(10, 5))

# Get top 3 and bottom 3 apps
top_apps = app_sentiment.nlargest(3)
bottom_apps = app_sentiment.nsmallest(3)
selected_apps = pd.concat([top_apps, bottom_apps])

colors5 = ['#2ecc71'] * 3 + ['#e74c3c'] * 3
bars5 = ax5.bar(range(len(selected_apps)), selected_apps.values, color=colors5)
ax5.set_xticks(range(len(selected_apps)))
ax5.set_xticklabels(selected_apps.index, fontsize=11, rotation=45, ha='right')
ax5.axhline(y=0, color='black', linewidth=0.5)
ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax5.set_ylabel('Average Sentiment Score', fontsize=12)
ax5.set_title('Highest and Lowest Performing Chatbots\nby User Sentiment', fontsize=14, fontweight='bold')
ax5.set_ylim(0, 0.8)

# Add value labels
for i, v in enumerate(selected_apps.values):
    ax5.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('figure5_top_bottom_apps.png', dpi=300, bbox_inches='tight')
print("✅ Figure 5 saved: 'figure5_top_bottom_apps.png'")

print("\n" + "=" * 60)
print("ALL FIGURES CREATED SUCCESSFULLY!")
print("=" * 60)
print("\nSaved files:")
print("  📊 figure1_sentiment_pie.png")
print("  📊 figure2_sentiment_by_rating.png")
print("  📊 figure3_app_sentiment_comparison.png")
print("  📊 figure4_sentiment_by_rating_stacked.png")
print("  📊 figure5_top_bottom_apps.png")

plt.show()