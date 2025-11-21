import time
import datetime
import json
import os
import hashlib
from mastodon import Mastodon, MastodonNotFoundError
from textblob import TextBlob
from elasticsearch import Elasticsearch, helpers

# ==========================================
# CONFIGURATION
# ==========================================

# Mastodon Config
ACCESS_TOKEN = 'zuG68yri_Fp4RxFro2QzRK4RRAp1bcmFwyRc5Rjb9Sw' 
# Elasticsearch Config
ES_HOST = "http://localhost:9200" 
ES_INDEX = "mastodon_ai_events"

# Scrape Config
WINDOW_DAYS = 7
MAX_PAGES_PER_TAG = 50

# ==========================================
# DATABASE SETUP
# ==========================================

def get_es_client():
    # If you have a cloud instance/password, add basic_auth=('user', 'pass') here
    return Elasticsearch(ES_HOST)

def init_index(es):
    """Creates the index with specific mappings if it doesn't exist."""
    if not es.indices.exists(index=ES_INDEX):
        mapping = {
            "mappings": {
                "properties": {
                    "event": {"type": "keyword"},
                    "instance": {"type": "keyword"},
                    "content": {"type": "text"},
                    "created_at": {"type": "date"},
                    "polarity": {"type": "float"},
                    "subjectivity": {"type": "float"},
                    "sentiment_class": {"type": "keyword"},
                    "hashtag_searched": {"type": "keyword"}
                }
            }
        }
        es.indices.create(index=ES_INDEX, body=mapping)
        print(f"✅ Created new index: {ES_INDEX}")
    else:
        print(f"ℹ️  Index {ES_INDEX} already exists. Appending data...")

# ==========================================
# UTILITIES
# ==========================================

def load_config_files():
    instances = []
    if os.path.exists('instances.txt'):
        with open('instances.txt', 'r') as f:
            instances = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        instances = ['https://mastodon.social', 'https://hachyderm.io']
        
    events = []
    if os.path.exists('ai_events.json'):
        with open('ai_events.json', 'r') as f:
            events = json.load(f)
            
    return instances, events

def date_to_snowflake(date_obj):
    return int(date_obj.timestamp() * 1000) << 16

def get_sentiment(text):
    clean_text = text.replace("<br>", " ").replace("<p>", "").replace("</p>", "")
    blob = TextBlob(clean_text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def classify_sentiment(polarity):
    if polarity > 0.05: return 'Positive'
    elif polarity < -0.05: return 'Negative'
    return 'Neutral'

# ==========================================
# CORE LOGIC
# ==========================================

def scrape_event(es, instance_url, event):
    print(f"\n--- Scraping {event['name']} on {instance_url} ---")
    
    target_date = datetime.datetime.strptime(event['date'], "%Y-%m-%d")
    start_date = target_date - datetime.timedelta(days=WINDOW_DAYS)
    end_date = target_date + datetime.timedelta(days=WINDOW_DAYS)
    max_id = date_to_snowflake(end_date)
    
    try:
        # Guest client (no token needed for public tags usually)
        client = Mastodon(api_base_url=instance_url)
    except:
        print(f"   ❌ Connection failed to {instance_url}")
        return

    total_scraped = 0
    
    for tag in event['hashtags']:
        print(f"   #{tag}...", end="", flush=True)
        current_max_id = max_id
        actions = [] # Buffer for bulk indexing
        seen_ids = set()
        
        for page in range(MAX_PAGES_PER_TAG):
            try:
                timeline = client.timeline_hashtag(tag, local=True, limit=40, max_id=current_max_id)
                if not timeline: break 
                
                current_max_id = timeline[-1]['id']
                
                for toot in timeline:
                    created_at = toot['created_at'].replace(tzinfo=None)
                    if created_at < start_date: continue
                    if toot['id'] in seen_ids: continue
                    
                    seen_ids.add(toot['id'])
                    pol, subj = get_sentiment(toot['content'])
                    
                    # Create a unique ID to prevent duplicates in ES
                    # Composite key: instance + post_id
                    doc_id = f"{instance_url}_{toot['id']}"
                    
                    doc = {
                        "_index": ES_INDEX,
                        "_id": doc_id, # Idempotency key
                        "_source": {
                            "event": event['name'],
                            "event_date": event['date'],
                            "instance": instance_url,
                            "content": toot['content'],
                            "created_at": toot['created_at'],
                            "polarity": pol,
                            "subjectivity": subj,
                            "sentiment_class": classify_sentiment(pol),
                            "hashtag_searched": tag
                        }
                    }
                    actions.append(doc)
                
                if timeline[-1]['created_at'].replace(tzinfo=None) < start_date:
                    break
                time.sleep(0.2)
                
            except MastodonNotFoundError: break
            except Exception: break
            
        # Bulk push to ES for this tag
        if actions:
            helpers.bulk(es, actions)
            total_scraped += len(actions)
            print(f" {len(actions)} indexed.", end="")
        else:
            print(" 0 found.", end="")
            
    print(f"\n   ✅ Total indexed for event: {total_scraped}")

# ==========================================
# MAIN
# ==========================================

def main():
    es = get_es_client()
    if not es.ping():
        print("❌ Could not connect to Elasticsearch. Is it running?")
        return

    init_index(es)
    instances, events = load_config_files()
    
    print(f"🚀 Starting Scraper: {len(events)} events x {len(instances)} instances")
    
    for event in events:
        for instance in instances:
            scrape_event(es, instance, event)
            
    print("\n🎉 Scraping Complete. Data is safely in Elasticsearch.")

if __name__ == "__main__":
    main()