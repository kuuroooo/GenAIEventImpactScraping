import json
import os
import sys
import argparse
import re
import numpy as np
from textblob import TextBlob

# --- NEW: Advanced NLP Imports ---
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import text2emotion as te
    EMOTION_AVAILABLE = True
except ImportError:
    EMOTION_AVAILABLE = False

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
    banner = f"""{Colors.GREEN}
   _____                   _           _     
  / ____|                 | |         (_)    
 | (___   __ _ _ __   __ _| |_   _ ___ _ ___ 
  \___ \ / _` | '_ \ / _` | | | | / __| / __|
  ____) | (_| | | | | (_| | | |_| \__ \ \__ \\
 |_____/ \__,_|_| |_|\__,_|_|\__, |___/_|___/
                              __/ |          
      {Colors.HEADER}:: SENTIMENT PROCESSOR ::{Colors.ENDC}    |___/           
    """
    print(banner)

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50):
    if total == 0: total = 1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{Colors.BLUE}{bar}{Colors.ENDC}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: print()

# ==========================================
# TOPIC MODELING ENGINE
# ==========================================

def apply_topic_modeling(data_list):
    """
    Applies Latent Dirichlet Allocation (LDA) per Event to find themes.
    """
    if not SKLEARN_AVAILABLE:
        return data_list

    print(f"\n{Colors.HEADER}=== RUNNING TOPIC MODELING (LDA) ==={Colors.ENDC}")
    
    # Group data by event to find topics specific to that event
    events = {}
    for doc in data_list:
        evt = doc.get('event', 'Unknown')
        if evt not in events: events[evt] = []
        events[evt].append(doc)

    total_events = len(events)
    processed_events = 0

    for event_name, docs in events.items():
        processed_events += 1
        print_progress_bar(processed_events, total_events, prefix='Modeling:', suffix=f'{event_name}')
        
        # Prepare corpus
        corpus = [d.get('content', '') for d in docs]
        if len(corpus) < 5: continue # Skip events with too little data

        # Vectorize (Remove stopwords, English only roughly)
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        try:
            tf = tf_vectorizer.fit_transform(corpus)
            feature_names = tf_vectorizer.get_feature_names_out()
        except ValueError:
            continue # Empty vocabulary or similar issue

        # Fit LDA (3 Topics per event is usually good for snapshots)
        n_topics = 3
        lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5, learning_method='online', random_state=0)
        lda.fit(tf)
        
        # Assign topics to documents
        topic_distributions = lda.transform(tf)
        
        # Generate readable topic labels (Top 3 words)
        topic_keywords = {}
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[:-4:-1]
            keywords = [feature_names[i] for i in top_indices]
            topic_keywords[topic_idx] = ", ".join(keywords)

        # Update original documents
        for i, doc in enumerate(docs):
            dominant_topic_idx = topic_distributions[i].argmax()
            doc['topic_id'] = int(dominant_topic_idx)
            doc['topic_keywords'] = topic_keywords[dominant_topic_idx]

    return data_list

# ==========================================
# SENTIMENT & EMOTION LOGIC
# ==========================================

def clean_text_content(text):
    clean = text.replace("<br>", " ").replace("<p>", "").replace("</p>", "")
    # Remove URLS
    clean = re.sub(r'http\S+', '', clean)
    return clean

def calculate_metrics(content):
    text = clean_text_content(content)
    
    # 1. TextBlob
    blob = TextBlob(text)
    tb_polarity = blob.sentiment.polarity
    tb_subjectivity = blob.sentiment.subjectivity
    
    # 2. VADER
    vader_score = 0.0
    if VADER_AVAILABLE:
        vader_score = vader_analyzer.polarity_scores(text)['compound']
        
    # 3. Emotion (Happy, Angry, Sad, Surprise, Fear)
    emotions = {}
    if EMOTION_AVAILABLE:
        # text2emotion returns dict: {'Happy': 0.0, 'Angry': 0.0, ...}
        # It can be slow, so we keep text short or accept the perf hit
        try:
            emotions = te.get_emotion(text)
        except:
            emotions = {'Happy': 0, 'Angry': 0, 'Surprise': 0, 'Sad': 0, 'Fear': 0}
            
    return tb_polarity, tb_subjectivity, vader_score, emotions

def classify_sentiment(polarity):
    if polarity > 0.05: return 'Positive'
    elif polarity < -0.05: return 'Negative'
    return 'Neutral'

# ==========================================
# MAIN PIPELINE
# ==========================================

def process_database(input_file, output_file, skip_emotion):
    if not os.path.exists(input_file):
        print(f"{Colors.FAIL}[!] Input file '{input_file}' not found.{Colors.ENDC}")
        return

    print(f"{Colors.CYAN}[INFO] Loading raw data into memory...{Colors.ENDC}")
    raw_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))
    
    total_lines = len(raw_data)
    print(f"   -> Loaded {Colors.BOLD}{total_lines}{Colors.ENDC} records.")

    print(f"\n{Colors.HEADER}=== RUNNING SENTIMENT & EMOTION ANALYSIS ==={Colors.ENDC}")
    
    models_msg = f"Models: {Colors.BOLD}TextBlob{Colors.ENDC}"
    if VADER_AVAILABLE: models_msg += f" + {Colors.BOLD}VADER{Colors.ENDC}"
    if EMOTION_AVAILABLE and not skip_emotion: models_msg += f" + {Colors.BOLD}Emotion (text2emotion){Colors.ENDC}"
    print(models_msg)
    
    processed_data = []
    
    for i, record in enumerate(raw_data):
        try:
            content = record.get('content', '')
            
            # Core Metrics
            pol, subj, vader, emotions = calculate_metrics(content)
            
            # Enrich Record
            record['polarity'] = pol
            record['subjectivity'] = subj
            record['vader_score'] = vader
            record['sentiment_class'] = classify_sentiment(pol)
            
            # Emotion (Flatten the dict)
            if emotions and not skip_emotion:
                for emo, val in emotions.items():
                    record[f'emotion_{emo.lower()}'] = val
            
            processed_data.append(record)
            
        except Exception:
            pass
        
        # Update UI
        if i % 10 == 0 or i == total_lines - 1:
            print_progress_bar(i + 1, total_lines, prefix='Processing:', suffix=f'{i+1}/{total_lines}')

    # --- TOPIC MODELING PHASE ---
    if SKLEARN_AVAILABLE:
        processed_data = apply_topic_modeling(processed_data)
    else:
        print(f"\n{Colors.WARNING}[WARN] scikit-learn not found. Skipping Topic Modeling.{Colors.ENDC}")

    # Save Results
    print(f"\n\n{Colors.CYAN}[INFO] Saving enriched data to '{output_file}'...{Colors.ENDC}")
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for entry in processed_data:
                json.dump(entry, outfile, default=str)
                outfile.write('\n')
        print(f"{Colors.GREEN}[SUCCESS] Processing complete!{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}[ERROR] Could not save file: {e}{Colors.ENDC}")

def main():
    print_banner()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='mastodon_raw.json', help="Raw scraper output")
    parser.add_argument('--output', type=str, default='mastodon_analyzed.json', help="Enriched database output")
    parser.add_argument('--no-emotion', action='store_true', help="Skip emotion analysis (faster)")
    args = parser.parse_args()
    
    # Dependency Checks
    if not VADER_AVAILABLE:
        print(f"{Colors.WARNING}[WARN] VADER library missing. (pip install vaderSentiment){Colors.ENDC}")
    if not EMOTION_AVAILABLE and not args.no_emotion:
        print(f"{Colors.WARNING}[WARN] text2emotion library missing. (pip install text2emotion){Colors.ENDC}")
    if not SKLEARN_AVAILABLE:
        print(f"{Colors.WARNING}[WARN] scikit-learn missing. Topic modeling disabled.{Colors.ENDC}")

    process_database(args.input, args.output, args.no_emotion)

if __name__ == "__main__":
    main()