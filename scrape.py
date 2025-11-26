import time
import datetime
import json
import os
import sys
import argparse
from mastodon import Mastodon, MastodonNotFoundError

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

def print_banner():
    banner = f"""{Colors.CYAN}
   _____                                  
  / ____|                                 
 | (___   ___ _ __ __ _ _ __   ___ _ __   
  \___ \ / __| '__/ _` | '_ \ / _ \ '__|  
  ____) | (__| | | (_| | |_) |  __/ |     
 |_____/ \___|_|  \__,_| .__/ \___|_|     
                       | |                
      {Colors.HEADER}:: RAW DATA COLLECTOR FOR MASTODON::{Colors.ENDC}      |_|                
    """
    print(banner)

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=40):
    if total == 0: total = 1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{Colors.BLUE}{bar}{Colors.ENDC}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: print()

# ==========================================
# CORE LOGIC
# ==========================================

def load_config_files(instances_path, events_path):
    instances = []
    if os.path.exists(instances_path):
        with open(instances_path, 'r') as f:
            instances = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        print(f"{Colors.FAIL}[!] Instances file '{instances_path}' not found.{Colors.ENDC}")
        
    events = []
    if os.path.exists(events_path):
        with open(events_path, 'r') as f:
            events = json.load(f)
    else:
        print(f"{Colors.FAIL}[!] Events file '{events_path}' not found.{Colors.ENDC}")
            
    return instances, events

def save_to_file(data_list, filepath):
    if not data_list: return
    with open(filepath, 'a', encoding='utf-8') as f:
        for entry in data_list:
            json.dump(entry, f, default=str)
            f.write('\n')

def scrape_event(instance_url, event, existing_ids, window_days, output_file, access_token=None):
    print(f"\n{Colors.BOLD}TARGET:{Colors.ENDC} {event['name']} {Colors.BOLD}@{Colors.ENDC} {instance_url}")
    
    target_date = datetime.datetime.strptime(event['date'], "%Y-%m-%d")
    start_date = target_date - datetime.timedelta(days=window_days)
    end_window = target_date + datetime.timedelta(days=window_days)
    max_id = int(end_window.timestamp() * 1000) << 16
    
    try:
        if access_token:
            client = Mastodon(api_base_url=instance_url, access_token=access_token, request_timeout=10)
        else:
            client = Mastodon(api_base_url=instance_url, request_timeout=10)
    except Exception as e:
        print(f"{Colors.FAIL}[ERROR] Connection failed: {e}{Colors.ENDC}")
        return

    total_scraped_for_event = 0
    max_pages = 50
    
    for tag in event['hashtags']:
        print(f"  > Scanning tag: {Colors.CYAN}#{tag}{Colors.ENDC}")
        current_max_id = max_id
        batch = []
        seen_ids = set()
        
        print_progress_bar(0, max_pages, prefix='    Progress:', suffix='Initializing...', length=30)
        
        for page in range(max_pages):
            try:
                timeline = client.timeline_hashtag(tag, local=True, limit=40, max_id=current_max_id)
                if not timeline:
                    print_progress_bar(max_pages, max_pages, prefix='    Progress:', suffix='Done (End of Feed)', length=30)
                    break 
                
                current_max_id = timeline[-1]['id']
                
                for toot in timeline:
                    created_at = toot['created_at']
                    if created_at.tzinfo: created_at = created_at.replace(tzinfo=None)
                    
                    if created_at < start_date: continue
                    if toot['id'] in seen_ids: continue
                    
                    unique_id = f"{instance_url}_{toot['id']}"
                    if unique_id in existing_ids: continue 
                    
                    seen_ids.add(toot['id'])
                    existing_ids.add(unique_id)
                    
                    # RAW DATA ONLY - NO SENTIMENT CALCULATION
                    doc = {
                        "id": unique_id,
                        "event": event['name'],
                        "event_date": event['date'],
                        "instance": instance_url,
                        "content": toot['content'],
                        "created_at": toot['created_at'],
                        "hashtag_searched": tag,
                        "url": toot['url']
                    }
                    batch.append(doc)
                
                last_ts = timeline[-1]['created_at']
                if last_ts.tzinfo: last_ts = last_ts.replace(tzinfo=None)
                if last_ts < start_date:
                    print_progress_bar(max_pages, max_pages, prefix='    Progress:', suffix='Done (Date Reached)', length=30)
                    break
                
                print_progress_bar(page + 1, max_pages, prefix='    Progress:', suffix=f"Found: {len(batch)}", length=30)
                time.sleep(0.2)
                
            except (MastodonNotFoundError, Exception):
                break
            
        if batch:
            save_to_file(batch, output_file)
            total_scraped_for_event += len(batch)
            sys.stdout.write(f"\r    {Colors.GREEN}[SUCCESS]{Colors.ENDC} Archived {len(batch)} raw posts          \n")
        else:
            sys.stdout.write(f"\r    {Colors.WARNING}[INFO]{Colors.ENDC} No new data found for #{tag}          \n")

# ==========================================
# MAIN
# ==========================================

def main():
    print_banner()
    parser = argparse.ArgumentParser()
    parser.add_argument('--token-file', type=str)
    parser.add_argument('--output', type=str, default='mastodon_raw.json', help="Raw data storage")
    parser.add_argument('--events', type=str, default='ai_events.json')
    parser.add_argument('--instances', type=str, default='instances.txt')
    parser.add_argument('--days', type=int, default=7)
    args = parser.parse_args()
    
    access_token = None
    if args.token_file and os.path.exists(args.token_file):
        with open(args.token_file, 'r') as f: access_token = f.read().strip()

    instances, events = load_config_files(args.instances, args.events)
    existing_ids = set()
    
    if os.path.exists(args.output):
        print(f"{Colors.CYAN}[INFO] Loading existing raw database...{Colors.ENDC}")
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip(): existing_ids.add(json.loads(line).get('id'))
            print(f"   -> Found {len(existing_ids)} records.")
        except: pass

    print(f"\n{Colors.HEADER}=== STARTING RAW SCRAPE ==={Colors.ENDC}")
    for event in events:
        for instance in instances:
            scrape_event(instance, event, existing_ids, args.days, args.output, access_token)

if __name__ == "__main__":
    main()