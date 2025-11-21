import time
import datetime
import pandas as pd
import json
import os
from mastodon import Mastodon, MastodonError, MastodonNotFoundError
from textblob import TextBlob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================

ACCESS_TOKEN = 'zuG68yri_Fp4RxFro2QzRK4RRAp1bcmFwyRc5Rjb9Sw' 
API_BASE_URL = 'https://mastodon.social' 

# Increased window to 1 week (7 days)
WINDOW_DAYS = 7
# Increased depth to ~1000 potential posts per tag
MAX_PAGES_PER_TAG = 50 

# ==========================================
# DATA LOADING
# ==========================================

def load_config_files():
    # Load Instances
    instances = []
    # Updated to match your filename 'instances.txt'
    if os.path.exists('instances.txt'):
        with open('instances.txt', 'r') as f:
            instances = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        # Fallback
        instances = ['https://mastodon.social', 'https://hachyderm.io']
        print("⚠️ 'instances.txt' not found. Using defaults.")

    # Load Events
    events = []
    if os.path.exists('ai_events.json'):
        with open('ai_events.json', 'r') as f:
            events = json.load(f)
    else:
        print("⚠️ 'ai_events.json' not found. Please create it.")
    
    return instances, events

# ==========================================
# UTILITIES
# ==========================================

def verify_api_access(access_token, api_base_url):
    print(f"--- Verifying Credentials for {api_base_url} ---")
    
    print("\n🔍 SENTIMENT ANALYSIS CONFIGURATION:")
    print("   Engine: TextBlob (Lexicon-based)")
    print("   Classification Logic:")
    print("     - Positive: Polarity > 0.05")
    print("     - Negative: Polarity < -0.05")
    print("     - Neutral:  -0.05 <= Polarity <= 0.05\n")

    if not access_token or access_token == 'YOUR_ACCESS_TOKEN_HERE':
        print("⚠️  WARNING: No ACCESS_TOKEN. Rate limits will be strict.\n")
        return False

    try:
        client = Mastodon(access_token=access_token, api_base_url=api_base_url)
        user = client.account_verify_credentials()
        try:
            version = client.retrieve_mastodon_version()
        except:
            version = client.instance().get('version', 'Unknown')

        print(f"✅ Success! Authenticated as: @{user['username']}")
        print(f"   Server version: {version}")
        print("------------------------------------------------\n")
        return True
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

def date_to_snowflake(date_obj):
    timestamp_ms = int(date_obj.timestamp() * 1000)
    return timestamp_ms << 16

def get_sentiment(text):
    clean_text = text.replace("<br>", " ").replace("<p>", "").replace("</p>", "")
    blob = TextBlob(clean_text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def classify_sentiment(polarity):
    if polarity > 0.05:
        return 'Positive'
    elif polarity < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# ==========================================
# CORE LOGIC
# ==========================================

def analyze_event(client, instance_url, event):
    print(f"\n--- Analyzing {event['name']} on {instance_url} ---")
    
    target_date = datetime.datetime.strptime(event['date'], "%Y-%m-%d")
    start_date = target_date - datetime.timedelta(days=WINDOW_DAYS)
    end_date = target_date + datetime.timedelta(days=WINDOW_DAYS)
    
    max_id = date_to_snowflake(end_date)
    
    try:
        instance_client = Mastodon(api_base_url=instance_url)
    except:
        print(f"Skipping {instance_url}: Connection failed.")
        return []

    relevant_toots = []
    seen_ids = set()

    for tag in event['hashtags']:
        print(f"   Searching #{tag}...", end="", flush=True)
        
        current_max_id = max_id
        found_for_tag = 0
        
        for page in range(MAX_PAGES_PER_TAG):
            try:
                timeline = instance_client.timeline_hashtag(tag, local=True, limit=40, max_id=current_max_id)
                
                if not timeline:
                    break 
                
                current_max_id = timeline[-1]['id']
                
                for toot in timeline:
                    created_at = toot['created_at'].replace(tzinfo=None)
                    
                    if created_at < start_date:
                        continue

                    if toot['id'] in seen_ids:
                        continue
                    
                    seen_ids.add(toot['id'])
                    found_for_tag += 1
                    
                    pol, subj = get_sentiment(toot['content'])
                    relevant_toots.append({
                        'event': event['name'],
                        'date_str': f"{event['name']}\n({event['date']})",
                        'instance': instance_url,
                        'created_at': toot['created_at'],
                        'polarity': pol,
                        'sentiment_class': classify_sentiment(pol),
                        'subjectivity': subj,
                        'content': toot['content']
                    })
                
                if timeline[-1]['created_at'].replace(tzinfo=None) < start_date:
                    break
                    
                time.sleep(0.2) 

            except MastodonNotFoundError:
                break 
            except Exception:
                break
        
        print(f" {found_for_tag} found.")

    print(f"   ✅ Unique Toots: {len(relevant_toots)}")
    return relevant_toots

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("Loading Configuration...")
    instances, events = load_config_files()
    
    if not events:
        return

    print(f"Starting Analysis for {len(events)} events across {len(instances)} instances.")
    verify_api_access(ACCESS_TOKEN, API_BASE_URL)
    
    all_data = []
    
    for event in events:
        for instance in instances:
            data = analyze_event(None, instance, event)
            all_data.extend(data)
            time.sleep(1) 

    if not all_data:
        print("No data found.")
        return

    df = pd.DataFrame(all_data)
    
    # --- Analysis Output ---
    print("\n=== RESULTS SUMMARY ===")
    if not df.empty:
        summary = df.groupby(['event', 'instance'])[['polarity', 'subjectivity']].mean()
        print(summary)

        # --- CHART 1: TIMELINE ANALYSIS (Events) ---
        plt.figure(figsize=(16, 10))
        
        # Subplot 1: Average Polarity
        plt.subplot(2, 1, 1)
        pivot_avg = df.groupby('date_str')['polarity'].mean().reindex([f"{e['name']}\n({e['date']})" for e in events])
        colors = ['#ff9999' if x < 0 else '#99ff99' for x in pivot_avg.values]
        pivot_avg.plot(kind='bar', color=colors, edgecolor='black', alpha=0.8)
        plt.title('Average Sentiment Polarity per Event', fontsize=14)
        plt.axhline(0, color='black', linewidth=1)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Subplot 2: Stacked Distribution
        plt.subplot(2, 1, 2)
        sentiment_counts = df.groupby(['date_str', 'sentiment_class']).size().unstack(fill_value=0)
        chronological_labels = [f"{e['name']}\n({e['date']})" for e in events]
        sentiment_counts = sentiment_counts.reindex(chronological_labels)
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
        plt.savefig('ai_sentiment_timeline.png')
        print("\nTimeline chart saved as 'ai_sentiment_timeline.png'")

        # --- CHART 2: PLATFORM BREAKDOWN (New Request) ---
        plt.figure(figsize=(14, 8))
        
        # Group by Instance and Sentiment
        # We calculate raw counts here for the "Number of posts" requirement
        platform_counts = df.groupby(['instance', 'sentiment_class']).size().unstack(fill_value=0)
        
        # Sort instances by total activity so the biggest ones are first
        platform_counts['Total'] = platform_counts.sum(axis=1)
        platform_counts = platform_counts.sort_values('Total', ascending=False).drop(columns='Total')
        
        # Ensure all sentiment columns exist
        for col in ['Negative', 'Neutral', 'Positive']:
            if col not in platform_counts:
                platform_counts[col] = 0
        
        # Plot Stacked Bar Chart
        ax = platform_counts[['Negative', 'Neutral', 'Positive']].plot(
            kind='bar',
            stacked=True,
            color=['#ff6666', '#dddddd', '#66cc66'],
            edgecolor='black',
            width=0.8,
            figsize=(14, 8)
        )
        
        plt.title('Total Posts & Sentiment Distribution per Platform', fontsize=16)
        plt.ylabel('Number of Posts', fontsize=12)
        plt.xlabel('Mastodon Instance', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Sentiment')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        # Add numeric labels inside the bars
        for c in ax.containers:
            # Only label if the segment is big enough to be readable
            labels = [v.get_height() if v.get_height() > 0 else '' for v in c]
            ax.bar_label(c, labels=labels, label_type='center', color='black', fontsize=9, padding=3)
            
        plt.savefig('platform_sentiment_breakdown.png')
        print("Platform Breakdown chart saved as 'platform_sentiment_breakdown.png'")

    else:
        print("Dataframe is empty.")

if __name__ == "__main__":
    main()