import pandas as pd
import matplotlib.pyplot as plt
import os
import json

# ==========================================
# CONFIGURATION
# ==========================================

DATA_FILE = "mastodon_data.json"

# ==========================================
# DATA LOADING
# ==========================================

def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"❌ File '{DATA_FILE}' not found. Run the scraper first!")
        return pd.DataFrame()

    print(f"⏳ Loading data from '{DATA_FILE}'...")
    
    data = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"✅ Loaded {len(data)} records.")
    return pd.DataFrame(data)

# ==========================================
# VISUALIZATION LOGIC
# ==========================================

def generate_visualizations(df):
    if df.empty:
        print("⚠️ DataFrame is empty.")
        return

    # Create display label
    df['label'] = df['event'] + "\n(" + df['event_date'] + ")"
    
    # Sort chronologically
    unique_events = df[['label', 'event_date']].drop_duplicates().sort_values('event_date')
    chronological_labels = unique_events['label'].tolist()

    print("📊 Generating Charts...")

    # --- CHART 1: TIMELINE ANALYSIS ---
    plt.figure(figsize=(16, 10))
    
    # Subplot 1: Average Polarity
    plt.subplot(2, 1, 1)
    pivot_avg = df.groupby('label')['polarity'].mean().reindex(chronological_labels)
    colors = ['#ff9999' if x < 0 else '#99ff99' for x in pivot_avg.values]
    pivot_avg.plot(kind='bar', color=colors, edgecolor='black', alpha=0.8)
    plt.title('Average Sentiment Polarity per Event', fontsize=14)
    plt.axhline(0, color='black', linewidth=1)
    plt.xticks([]) # Hide x labels for top plot

    # Subplot 2: Sentiment Distribution (Stacked)
    plt.subplot(2, 1, 2)
    sentiment_counts = df.groupby(['label', 'sentiment_class']).size().unstack(fill_value=0)
    sentiment_counts = sentiment_counts.reindex(chronological_labels)
    
    # Convert to percentages
    sentiment_pct = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100
    
    for col in ['Negative', 'Neutral', 'Positive']:
        if col not in sentiment_pct: sentiment_pct[col] = 0
        
    sentiment_pct[['Negative', 'Neutral', 'Positive']].plot(
        kind='bar', stacked=True, color=['#ff6666', '#dddddd', '#66cc66'], 
        ax=plt.gca(), edgecolor='black', width=0.8
    )
    plt.title('Community Response Distribution (Positive vs Negative)', fontsize=14)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('timeline_analysis.png')
    print("   -> Saved 'timeline_analysis.png'")

    # --- CHART 2: PLATFORM BREAKDOWN ---
    plt.figure(figsize=(14, 8))
    
    platform_counts = df.groupby(['instance', 'sentiment_class']).size().unstack(fill_value=0)
    platform_counts['Total'] = platform_counts.sum(axis=1)
    platform_counts = platform_counts.sort_values('Total', ascending=False).drop(columns='Total')
    
    for col in ['Negative', 'Neutral', 'Positive']:
        if col not in platform_counts: platform_counts[col] = 0
        
    ax = platform_counts[['Negative', 'Neutral', 'Positive']].plot(
        kind='bar', stacked=True, color=['#ff6666', '#dddddd', '#66cc66'],
        edgecolor='black', width=0.8, figsize=(14,8)
    )
    
    plt.title('Total Posts & Sentiment per Platform (All Events)', fontsize=16)
    plt.xlabel('Instance', fontsize=12)
    plt.ylabel('Number of Posts', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add number labels
    for c in ax.containers:
        labels = [int(v.get_height()) if v.get_height() > 0 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=9)
        
    plt.savefig('platform_breakdown.png')
    print("   -> Saved 'platform_breakdown.png'")

def main():
    df = load_data()
    generate_visualizations(df)

if __name__ == "__main__":
    main()