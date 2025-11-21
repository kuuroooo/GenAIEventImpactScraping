import pandas as pd
import matplotlib.pyplot as plt
from elasticsearch import Elasticsearch, helpers
import os

# ==========================================
# CONFIGURATION
# ==========================================

ES_HOST = "http://localhost:9200"
ES_INDEX = "mastodon_ai_events"

# ==========================================
# DATA FETCHING
# ==========================================

def fetch_data_from_es():
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        print("❌ Could not connect to Elasticsearch.")
        return pd.DataFrame()

    print(f"⏳ Fetching data from index '{ES_INDEX}'...")
    
    # "scan" is a helper that handles pagination (scrolling) for large datasets
    # We fetch everything matching our index
    query = {"query": {"match_all": {}}}
    
    scan_gen = helpers.scan(
        es,
        query=query,
        index=ES_INDEX,
        scroll='2m'
    )
    
    # Convert the generator to a list of dicts, then to DataFrame
    data = [doc['_source'] for doc in scan_gen]
    
    print(f"✅ Loaded {len(data)} records from database.")
    return pd.DataFrame(data)

# ==========================================
# VISUALIZATION LOGIC
# ==========================================

def generate_visualizations(df):
    if df.empty:
        print("⚠️ DataFrame is empty. Run the scraper first!")
        return

    # Create a helper column for nice labels
    # Combine Event Name + Date
    df['label'] = df['event'] + "\n(" + df['event_date'] + ")"
    
    # 1. Sort events chronologically
    # We sort by the actual 'event_date' string to ensure order is correct
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
    
    # Ensure cols exist
    for col in ['Negative', 'Neutral', 'Positive']:
        if col not in sentiment_pct: sentiment_pct[col] = 0
        
    sentiment_pct[['Negative', 'Neutral', 'Positive']].plot(
        kind='bar', stacked=True, color=['#ff6666', '#dddddd', '#66cc66'], 
        ax=plt.gca(), edgecolor='black', width=0.8
    )
    plt.title('Community Response Distribution (Positive vs Negative)', fontsize=14)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('es_timeline_analysis.png')
    print("   -> Saved 'es_timeline_analysis.png'")

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
    
    # Add labels
    for c in ax.containers:
        labels = [int(v.get_height()) if v.get_height() > 0 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=9)
        
    plt.savefig('es_platform_breakdown.png')
    print("   -> Saved 'es_platform_breakdown.png'")

def main():
    df = fetch_data_from_es()
    generate_visualizations(df)

if __name__ == "__main__":
    main()