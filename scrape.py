import time
import datetime
import json
import os
import sys
from mastodon import Mastodon, MastodonNotFoundError
from textblob import TextBlob

# ==========================================
# VISUAL CONFIGURATION
# ==========================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_banner():
    banner = f"""{Colors.CYAN}
  __  __           _            _             
 |  \/  |         | |          | |            
 | \  / | __ _ ___| |_ ___   __| | ___  _ __  
 | |\/| |/ _` / __| __/ _ \ / _` |/ _ \| '_ \ 
 | |  | | (_| \__ \ || (_) | (_| | (_) | | | |
 |_|  |_|\__,_|___/\__\___/ \__,_|\___/|_| |_|
                                              
      {Colors.HEADER}:: AI EVENT IMPACT ANALYZER ::{Colors.ENDC}
    """
    print(banner)

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=40, fill='='):
    """
    Call in a loop to create terminal progress bar
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    # Clear line before printing to keep it on same line
    sys.stdout.write(f'\r{prefix} |{Colors.BLUE}{bar}{Colors.ENDC}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()

# ==========================================
# CONFIGURATION
# ==========================================

DATA_FILE = "mastodon_data.json"
WINDOW_DAYS = 7
MAX_PAGES_PER_TAG = 50

# ==========================================
# UTILITIES
# ==========================================

def load_config_files():
    instances = []
    if os.path.exists('instances.txt'):
        with open('instances.txt', 'r') as f:
            instances = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        print(f"{Colors.FAIL}[!] instances.txt not found.{Colors.ENDC}")
        return [], []
        
    events = []
    if os.path.exists('ai_events.json'):
        with open('ai_events.json', 'r') as f:
            events = json.load(f)
            
    return instances, events

def get_sentiment(text):
    clean_text = text.replace("<br>", " ").replace("<p>", "").replace("</p>", "")
    blob = TextBlob(clean_text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def classify_sentiment(polarity):
    if polarity > 0.05: return 'Positive'
    elif polarity < -0.05: return 'Negative'
    return 'Neutral'

def save_to_file(data_list):
    """Appends a list of dictionaries to the JSON file (One JSON object per line)"""
    if not data_list:
        return
        
    with open(DATA_FILE, 'a', encoding='utf-8') as f:
        for entry in data_list:
            json.dump(entry, f, default=str)
            f.write('\n') # Newline delimiter

# ==========================================
# CORE LOGIC
# ==========================================

def scrape_event(instance_url, event, existing_ids):
    # Print Header for this specific scrape task
    print(f"\n{Colors.BOLD}TARGET:{Colors.ENDC} {event['name']} {Colors.BOLD}@{Colors.ENDC} {instance_url}")
    
    target_date = datetime.datetime.strptime(event['date'], "%Y-%m-%d")
    start_date = target_date - datetime.timedelta(days=WINDOW_DAYS)
    end_date = target_date + datetime.timedelta(days=WINDOW_DAYS)
    
    # Snowflake ID Calculation
    max_id = int(end_date.timestamp() * 1000) << 16
    
    try:
        client = Mastodon(api_base_url=instance_url)
    except:
        print(f"{Colors.FAIL}[ERROR] Connection failed to {instance_url}{Colors.ENDC}")
        return

    total_scraped_for_event = 0
    
    for tag in event['hashtags']:
        print(f"  > Scanning tag: {Colors.CYAN}#{tag}{Colors.ENDC}")
        
        current_max_id = max_id
        batch = []
        seen_ids = set()
        
        # Run progress bar loop
        print_progress_bar(0, MAX_PAGES_PER_TAG, prefix='    Progress:', suffix='Initializing...', length=30)
        
        for page in range(MAX_PAGES_PER_TAG):
            try:
                timeline = client.timeline_hashtag(tag, local=True, limit=40, max_id=current_max_id)
                
                if not timeline:
                    # If no more posts, fill bar to 100% and break
                    print_progress_bar(MAX_PAGES_PER_TAG, MAX_PAGES_PER_TAG, prefix='    Progress:', suffix=f'Done (End of Feed)', length=30)
                    break 
                
                current_max_id = timeline[-1]['id']
                
                for toot in timeline:
                    created_at = toot['created_at'].replace(tzinfo=None)
                    if created_at < start_date: continue
                    if toot['id'] in seen_ids: continue
                    
                    # Unique ID check (Instance + ID)
                    unique_id = f"{instance_url}_{toot['id']}"
                    if unique_id in existing_ids: continue # Skip duplicates
                    
                    seen_ids.add(toot['id'])
                    existing_ids.add(unique_id)
                    
                    pol, subj = get_sentiment(toot['content'])
                    
                    doc = {
                        "id": unique_id,
                        "event": event['name'],
                        "event_date": event['date'],
                        "instance": instance_url,
                        "content": toot['content'],
                        "created_at": toot['created_at'],
                        "polarity": pol,
                        "subjectivity": subj,
                        "sentiment_class": classify_sentiment(pol),
                        "hashtag_searched": tag,
                        "url": toot['url']
                    }
                    batch.append(doc)
                
                # Check if we went too far back in time
                if timeline[-1]['created_at'].replace(tzinfo=None) < start_date:
                    print_progress_bar(MAX_PAGES_PER_TAG, MAX_PAGES_PER_TAG, prefix='    Progress:', suffix=f'Done (Date Reached)', length=30)
                    break
                
                # Update Progress Bar
                suffix_text = f"Found: {len(batch)}"
                print_progress_bar(page + 1, MAX_PAGES_PER_TAG, prefix='    Progress:', suffix=suffix_text, length=30)
                
                time.sleep(0.2)
                
            except MastodonNotFoundError:
                print_progress_bar(MAX_PAGES_PER_TAG, MAX_PAGES_PER_TAG, prefix='    Progress:', suffix='Tag Not Found', length=30)
                break
            except Exception:
                break
            
        if batch:
            save_to_file(batch)
            total_scraped_for_event += len(batch)
            # Overwrite the progress bar line with a clean summary
            sys.stdout.write(f"\r    {Colors.GREEN}[SUCCESS]{Colors.ENDC} Saved {len(batch)} new posts for #{tag}          \n")
        else:
            sys.stdout.write(f"\r    {Colors.WARNING}[INFO]{Colors.ENDC} No new relevant data found for #{tag}          \n")
            
    print(f"{Colors.GREEN}>> Event Complete.{Colors.ENDC} Total captured: {total_scraped_for_event}")

# ==========================================
# MAIN
# ==========================================

def main():
    print_banner()
    instances, events = load_config_files()
    
    # Load existing IDs to prevent duplicates
    existing_ids = set()
    if os.path.exists(DATA_FILE):
        print(f"{Colors.CYAN}[INFO] Reading existing database: {DATA_FILE}...{Colors.ENDC}")
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        existing_ids.add(doc.get('id'))
            print(f"   -> Database contains {Colors.BOLD}{len(existing_ids)}{Colors.ENDC} records.")
        except Exception as e:
            print(f"   {Colors.FAIL}[!] Error reading file (starting fresh): {e}{Colors.ENDC}")

    print(f"\n{Colors.HEADER}=== INITIALIZING SCRAPER ==={Colors.ENDC}")
    print(f"Targets: {Colors.BOLD}{len(events)}{Colors.ENDC} Events x {Colors.BOLD}{len(instances)}{Colors.ENDC} Instances")
    print(f"Window:  {Colors.BOLD}{WINDOW_DAYS}{Colors.ENDC} days +/- event date")
    print("-" * 50)
    
    for event in events:
        for instance in instances:
            scrape_event(instance, event, existing_ids)
            
    print(f"\n{Colors.GREEN}=========================================={Colors.ENDC}")
    print(f"{Colors.GREEN}   MISSION ACCOMPLISHED{Colors.ENDC}")
    print(f"{Colors.GREEN}=========================================={Colors.ENDC}")
    print(f"Data saved to: {Colors.BOLD}{DATA_FILE}{Colors.ENDC}")

if __name__ == "__main__":
    main()