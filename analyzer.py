import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import sys
import argparse
import re
import numpy as np
from collections import Counter
from datetime import datetime
from math import pi, ceil

# Optional: NetworkX for Influencer Graph
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

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

STOPWORDS = set([
    'the', 'and', 'to', 'of', 'a', 'in', 'is', 'for', 'on', 'that', 'it', 'with', 'as', 'are', 
    'this', 'was', 'by', 'an', 'be', 'or', 'at', 'from', 'not', 'have', 'has', 'but', 'can', 
    'more', 'about', 'we', 'my', 'they', 'what', 'so', 'like', 'just', 'https', 'http', 'com', 
    'www', 'mastodon', 'social', 'content', 'html', 'href', 'rel', 'nofollow', 'target', 'blank',
    'span', 'class', 'br', 'p', 'div', 'label', 'translate', 'search', 'status', 'card', 'if', 
    'you', 'me', 'your', 'will', 'one', 'all', 'do', 'no', 'up', 'out', 'there', 'get', 'how',
    'when', 'some', 'time', 'now', 'only', 'new', 'amp', 'gt', 'lt', 'quot', 'people', 'ai'
])

def print_banner():
    banner = f"""{Colors.CYAN}
   _____                 _         
  / ____|               | |        
 | |  __ _ __ __ _ _ __ | |__  ___ 
 | | |_ | '__/ _` | '_ \| '_ \/ __|
 | |__| | | | (_| | |_) | | | \__ \\
  \_____|_|  \__,_| .__/|_| |_|___/
                  | |              
      {Colors.HEADER}:: VISUALYZER PRO ::{Colors.ENDC}      |_|              
    """
    print(banner)

# ==========================================
# DATA HELPERS
# ==========================================

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"{Colors.FAIL}[!] File '{filepath}' not found.{Colors.ENDC}")
        return pd.DataFrame()

    print(f"{Colors.CYAN}[INFO] Loading analyzed data from '{filepath}'...{Colors.ENDC}")
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        print(f"{Colors.FAIL}[ERROR] Corrupt data: {e}{Colors.ENDC}")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    if not df.empty:
        df['created_at'] = pd.to_datetime(df['created_at'], format='mixed', utc=True)
        df['event_date'] = pd.to_datetime(df['event_date'], format='mixed', utc=True)
        df['label'] = df['event'] + "\n(" + df['event_date'].dt.strftime('%Y-%m-%d') + ")"
    
    print(f"   -> Loaded {Colors.BOLD}{len(df)}{Colors.ENDC} records.")
    return df

def extract_keywords(text_series, top_n=10):
    all_words = []
    for text in text_series:
        if not isinstance(text, str): continue
        clean = re.sub(r'<[^>]+>', '', text).lower()
        clean = re.sub(r'[^\w\s]', '', clean)
        words = clean.split()
        all_words.extend([w for w in words if w not in STOPWORDS and len(w) > 3])
    return Counter(all_words).most_common(top_n)

def extract_emojis(text_series, top_n=10):
    emoji_pattern = re.compile(r'[\U0001F000-\U0001F9FF]|[\u2700-\u27BF]|[\u2600-\u26FF]')
    all_emojis = []
    for text in text_series:
        if not isinstance(text, str): continue
        found = emoji_pattern.findall(text)
        all_emojis.extend(found)
    return Counter(all_emojis).most_common(top_n)

def extract_mentions(text):
    if not isinstance(text, str): return []
    return re.findall(r'@([\w\d_]+)', text)

# ==========================================
# PLOTTING ENGINE
# ==========================================

def generate_visualizations(df, output_folder="figures"):
    if df.empty: return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"{Colors.CYAN}[INFO] Created output directory: {output_folder}{Colors.ENDC}")

    unique_events = df[['label', 'event_date', 'event']].drop_duplicates().sort_values('event_date')
    chronological_labels = unique_events['label'].tolist()
    event_names = unique_events['event'].tolist()
    
    print(f"\n{Colors.HEADER}=== GENERATING COMPREHENSIVE REPORT ==={Colors.ENDC}")

    # --- 1. TIMELINE SENTIMENT ---
    print(f"   > [1/15] Timeline Trends...", end=" ")
    plt.figure(figsize=(16, 10))
    
    plt.subplot(2, 1, 1)
    pivot_avg = df.groupby('label')['polarity'].mean().reindex(chronological_labels)
    colors = ['#ff4d4d' if x < 0 else '#4dff88' for x in pivot_avg.values]
    pivot_avg.plot(kind='bar', color=colors, edgecolor='black', alpha=0.8)
    plt.title('Average Sentiment Polarity per Event', fontsize=14, fontweight='bold')
    plt.axhline(0, color='black', linewidth=1)
    plt.xticks([]) 
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.subplot(2, 1, 2)
    sentiment_counts = df.groupby(['label', 'sentiment_class']).size().unstack(fill_value=0)
    sentiment_counts = sentiment_counts.reindex(chronological_labels)
    sentiment_pct = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100
    for col in ['Negative', 'Neutral', 'Positive']:
        if col not in sentiment_pct: sentiment_pct[col] = 0
        
    sentiment_pct[['Negative', 'Neutral', 'Positive']].plot(
        kind='bar', stacked=True, color=['#ff6666', '#e0e0e0', '#66cc66'], 
        ax=plt.gca(), edgecolor='black', width=0.8
    )
    plt.title('Community Reaction Split (Controversy Index)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "01_timeline_sentiment.png"))
    plt.close()
    print(f"{Colors.GREEN}[DONE]{Colors.ENDC}")

    # --- 2. SHARE OF VOICE ---
    print(f"   > [2/15] Share of Voice...", end=" ")
    plt.figure(figsize=(10, 8))
    event_counts = df['event'].value_counts()
    if len(event_counts) > 10:
        top_events = event_counts.head(9)
        other_count = event_counts.iloc[9:].sum()
        top_events['Other'] = other_count
        event_counts = top_events
    plt.pie(event_counts, labels=event_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title('Share of Voice: Dominant Events', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_folder, "02_share_of_voice.png"))
    plt.close()
    print(f"{Colors.GREEN}[DONE]{Colors.ENDC}")

    # --- 3. VOLUME SPIKES ---
    print(f"   > [3/15] Volume Spikes...", end=" ")
    plt.figure(figsize=(12, 6))
    vol_df = df.copy()
    vol_df['days_relative'] = (vol_df['created_at'] - vol_df['event_date']).dt.days
    vol_df = vol_df[(vol_df['days_relative'] >= -2) & (vol_df['days_relative'] <= 7)]
    pivot_vol = vol_df.groupby(['days_relative', 'event']).size().unstack(fill_value=0)
    pivot_vol.plot(kind='line', linewidth=2, marker='o', figsize=(12, 6))
    plt.title('Hype Cycle: Discussion Volume Relative to Launch Day', fontsize=14, fontweight='bold')
    plt.xlabel('Days Relative to Event (0 = Launch Day)', fontsize=12)
    plt.ylabel('Post Volume', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "03_volume_spikes.png"))
    plt.close()
    print(f"{Colors.GREEN}[DONE]{Colors.ENDC}")

    # --- 4. WEEKEND EFFECT ---
    print(f"   > [4/15] Weekend Effect Heatmap...", end=" ")
    plt.figure(figsize=(12, 6))
    time_df = df.copy()
    time_df['day_name'] = time_df['created_at'].dt.day_name()
    time_df['hour'] = time_df['created_at'].dt.hour
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_df['day_name'] = pd.Categorical(time_df['day_name'], categories=days_order, ordered=True)
    pivot_time = time_df.groupby(['day_name', 'hour'], observed=False).size().unstack(fill_value=0)
    sns.heatmap(pivot_time, cmap="YlGnBu", cbar_kws={'label': 'Post Volume'})
    plt.title('The "Weekend Effect": Activity Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "04_weekend_effect.png"))
    plt.close()
    print(f"{Colors.GREEN}[DONE]{Colors.ENDC}")

    # --- 5. SENTIMENT VELOCITY ---
    print(f"   > [5/15] Sentiment Velocity...", end=" ")
    plt.figure(figsize=(12, 6))
    pivot_velocity = vol_df.groupby(['days_relative', 'event'])['polarity'].mean().unstack()
    pivot_velocity.plot(linewidth=2, figsize=(12, 6), alpha=0.8)
    plt.title('Sentiment Velocity: Post-Launch Shifts', fontsize=14, fontweight='bold')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "05_sentiment_velocity.png"))
    plt.close()
    print(f"{Colors.GREEN}[DONE]{Colors.ENDC}")

    # --- 6. SCATTER MATRIX ---
    print(f"   > [6/15] Hot Take Matrix...", end=" ")
    plt.figure(figsize=(10, 8))
    scatter_data = df.groupby('event')[['polarity', 'subjectivity']].mean().reset_index()
    sns.scatterplot(data=scatter_data, x='polarity', y='subjectivity', s=200, hue='event', style='event')
    for i in range(scatter_data.shape[0]):
        plt.text(scatter_data.polarity[i]+0.01, scatter_data.subjectivity[i], scatter_data.event[i], fontsize=9)
    plt.axvline(0, color='gray', linestyle='--')
    plt.axhline(0.5, color='gray', linestyle='--')
    plt.title('Fact vs. Opinion Matrix', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "06_hot_take_matrix.png"))
    plt.close()
    print(f"{Colors.GREEN}[DONE]{Colors.ENDC}")

    # --- 7. INSTANCE LANDSCAPE ---
    print(f"   > [7/15] Instance Density...", end=" ")
    plt.figure(figsize=(12, 8))
    inst_stats = df.groupby('instance').agg({'polarity': 'mean', 'subjectivity': 'mean', 'id': 'count'}).reset_index()
    inst_stats = inst_stats[inst_stats['id'] > 10]
    sns.scatterplot(data=inst_stats, x='subjectivity', y='polarity', size='id', sizes=(50, 1000), alpha=0.6, color='purple', legend=False)
    top_insts = inst_stats.nlargest(5, 'id')
    for _, row in top_insts.iterrows():
        plt.text(row['subjectivity'], row['polarity'], row['instance'].split('//')[-1], fontsize=10, fontweight='bold')
    plt.title('Instance Landscape', fontsize=14, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "07_instance_density.png"))
    plt.close()
    print(f"{Colors.GREEN}[DONE]{Colors.ENDC}")

    # --- 8. POST LENGTH ---
    print(f"   > [8/15] Post Length...", end=" ")
    plt.figure(figsize=(10, 6))
    df['length'] = df['content'].astype(str).apply(len)
    sns.histplot(data=df, x='length', hue='sentiment_class', element="step", stat="density", common_norm=False)
    plt.xlim(0, 1000)
    plt.title('Post Length by Sentiment', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "08_post_length.png"))
    plt.close()
    print(f"{Colors.GREEN}[DONE]{Colors.ENDC}")

    # --- 9. EMOJI HEATMAP ---
    print(f"   > [9/15] Emoji Analysis...", end=" ")
    all_emojis = extract_emojis(df['content'], top_n=50)
    if all_emojis:
        top_10_emojis = [e[0] for e in all_emojis[:10]]
        emoji_matrix = []
        for evt in unique_events['event'].unique():
            evt_text = " ".join(df[df['event'] == evt]['content'].astype(str))
            counts = []
            for emo in top_10_emojis:
                counts.append(evt_text.count(emo))
            emoji_matrix.append(counts)
        plt.figure(figsize=(10, 8))
        sns.heatmap(emoji_matrix, annot=True, fmt="d", xticklabels=top_10_emojis, yticklabels=unique_events['event'].unique(), cmap="Oranges")
        plt.title('Emoji Usage Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "09_emoji_heatmap.png"))
        plt.close()
    print(f"{Colors.GREEN}[DONE]{Colors.ENDC}")

    # --- 10. KEYWORD CLOUDS ---
    print(f"   > [10/15] Keyword Deep Dives...")
    for evt in unique_events['event'].unique():
        evt_df = df[df['event'] == evt]
        if len(evt_df) < 5: continue
        pos_text = evt_df[evt_df['sentiment_class'] == 'Positive']['content']
        neg_text = evt_df[evt_df['sentiment_class'] == 'Negative']['content']
        pos_keywords = extract_keywords(pos_text)
        neg_keywords = extract_keywords(neg_text)
        if not pos_keywords and not neg_keywords: continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Context Analysis: {evt}', fontsize=16, fontweight='bold')
        if pos_keywords:
            words, counts = zip(*pos_keywords)
            ax1.barh(words, counts, color='#66cc66', edgecolor='black')
            ax1.set_title('Positive Drivers')
            ax1.invert_yaxis()
        if neg_keywords:
            words, counts = zip(*neg_keywords)
            ax2.barh(words, counts, color='#ff6666', edgecolor='black')
            ax2.set_title('Negative Drivers')
            ax2.invert_yaxis()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        safe_name = "".join(x for x in evt if x.isalnum())
        plt.savefig(os.path.join(output_folder, f"kw_{safe_name}.png"))
        plt.close()
    print(f"{Colors.GREEN}   > Keyword Analysis Complete.{Colors.ENDC}")

    # --- 11. BEFORE VS AFTER SHIFT (RAW COUNTS) ---
    print(f"   > [11/15] Volume Shift...", end=" ")
    df['period'] = np.where(df['created_at'] < df['event_date'], 'Before', 'After')
    ba_counts = df.groupby(['label', 'period', 'sentiment_class']).size().unstack(fill_value=0)
    for col in ['Negative', 'Neutral', 'Positive']:
        if col not in ba_counts: ba_counts[col] = 0
    
    def get_sort_key(index_tuple):
        lbl, per = index_tuple
        try:
            date_str = lbl.split('(')[-1].replace(')', '')
            dt = datetime.strptime(date_str, '%Y-%m-%d')
        except:
            dt = datetime.min
        period_rank = 0 if per == 'Before' else 1
        return (dt, lbl, period_rank)

    sorted_idx = sorted(ba_counts.index, key=get_sort_key)
    ba_counts = ba_counts.reindex(sorted_idx)
    plot_labels = [f"{idx[0].split()[0]}...\n({idx[1]})" for idx in ba_counts.index]
    
    plt.figure(figsize=(18, 8))
    ba_counts[['Negative', 'Neutral', 'Positive']].plot(
        kind='bar', stacked=True, color=['#ff6666', '#e0e0e0', '#66cc66'], 
        edgecolor='black', width=0.8, figsize=(18, 8)
    )
    plt.title('Volume & Sentiment Shift: Before vs. After', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Posts', fontsize=12)
    plt.xticks(range(len(plot_labels)), plot_labels, rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "11_before_after_shift.png"))
    plt.close()
    print(f"{Colors.GREEN}[DONE]{Colors.ENDC}")

    # --- 12. EMOTION RADAR CHART (PER EVENT GRID) ---
    print(f"   > [12/15] Emotion Radar (Multiples)...", end=" ")
    emotion_cols = [c for c in df.columns if c.startswith('emotion_')]
    if emotion_cols:
        # Aggregate emotions by event
        emo_means = df.groupby('event')[emotion_cols].mean()
        
        # Filter for events that actually exist in the data
        valid_events = [e for e in event_names if e in emo_means.index]
        
        if valid_events:
            categories = [c.replace('emotion_', '').capitalize() for c in emotion_cols]
            N = len(categories)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += [angles[0]] # Close loop
            
            # Grid calculation
            num_events = len(valid_events)
            cols = 3
            rows = ceil(num_events / cols)
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5), subplot_kw={'projection': 'polar'})
            axes = axes.flatten() if num_events > 1 else [axes]
            
            for idx, evt in enumerate(valid_events):
                ax = axes[idx]
                values = emo_means.loc[evt].values.tolist()
                values += [values[0]]
                
                # Plot
                ax.plot(angles, values, linewidth=2, linestyle='solid', color='blue')
                ax.fill(angles, values, color='blue', alpha=0.25)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories, fontsize=8)
                ax.set_title(evt, size=12, fontweight='bold', y=1.1)
                ax.grid(True)
            
            # Turn off unused subplots
            for i in range(num_events, len(axes)):
                axes[i].axis('off')
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, "12_emotion_radar.png"))
            print(f"{Colors.GREEN}[DONE]{Colors.ENDC}")
        else:
            print(f"{Colors.WARNING}[SKIP] No valid event data.{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}[SKIP] No emotion data found.{Colors.ENDC}")

    # --- 13. TOPIC DIVERSITY (PER EVENT BREAKDOWN) ---
    print(f"   > [13/15] Topic Distribution...", end=" ")
    if 'topic_keywords' in df.columns:
        # We want a breakdown of topics PER event.
        # Best visualized as a set of Horizontal Bar Charts
        
        # Filter for events
        valid_events = [e for e in event_names if not df[df['event'] == e]['topic_keywords'].isna().all()]
        
        if valid_events:
            num_events = len(valid_events)
            cols = 2
            rows = ceil(num_events / cols)
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols*8, rows*4))
            axes = axes.flatten() if num_events > 1 else [axes]
            
            for idx, evt in enumerate(valid_events):
                ax = axes[idx]
                evt_data = df[df['event'] == evt]
                topic_counts = evt_data['topic_keywords'].value_counts().head(5) # Top 5 topics per event
                
                # Shorten keywords for display (first 3 words)
                labels = [", ".join(k.split(', ')[:3]) for k in topic_counts.index]
                
                ax.barh(labels, topic_counts.values, color='teal', edgecolor='black', alpha=0.7)
                ax.set_title(evt, fontsize=12, fontweight='bold')
                ax.invert_yaxis() # Top topic at top
                
            # Turn off unused
            for i in range(num_events, len(axes)):
                axes[i].axis('off')
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, "13_topic_distribution.png"))
            print(f"{Colors.GREEN}[DONE]{Colors.ENDC}")
        else:
             print(f"{Colors.WARNING}[SKIP] No topic data.{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}[SKIP] No topic data found.{Colors.ENDC}")

    # --- 14. INFLUENCER NETWORK ---
    print(f"   > [14/15] Influencer Analysis...", end=" ")
    mentions = []
    for text in df['content']:
        extracted = extract_mentions(str(text))
        mentions.extend(extracted)
    
    if mentions:
        top_mentions = Counter(mentions).most_common(20)
        users, counts = zip(*top_mentions)
        
        if NETWORKX_AVAILABLE:
            G = nx.Graph()
            for user, count in top_mentions:
                G.add_node(user, size=count)
                G.add_edge("Community", user, weight=count)
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(G, k=0.5)
            sizes = [G.nodes[n].get('size', 10)*50 for n in G.nodes]
            nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='skyblue', alpha=0.7)
            nx.draw_networkx_edges(G, pos, width=1, alpha=0.3)
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
            plt.title('Top Mentioned Accounts (Influencer Map)', fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.savefig(os.path.join(output_folder, "14_influencer_network.png"))
            plt.close()
        else:
            plt.figure(figsize=(12, 8))
            plt.barh(users, counts, color='skyblue', edgecolor='black')
            plt.title('Top Mentioned Accounts', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, "14_influencer_bar.png"))
            plt.close()
        print(f"{Colors.GREEN}[DONE]{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}[SKIP] No mentions found.{Colors.ENDC}")

    # --- 15. GEOGRAPHIC/TIME-ZONE INFERENCE ---
    print(f"   > [15/15] Time-Zone Inference...", end=" ")
    hours = df['created_at'].dt.hour
    plt.figure(figsize=(12, 6))
    sns.histplot(hours, bins=24, kde=True, color='teal', edgecolor='black')
    plt.title('Global Activity: Posting Hour Distribution (UTC)', fontsize=14, fontweight='bold')
    plt.xlabel('Hour of Day (UTC)', fontsize=12)
    plt.xticks(range(0, 25, 2))
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.axvspan(8, 17, color='yellow', alpha=0.1, label='EU/Africa Working Hours')
    plt.axvspan(14, 23, color='blue', alpha=0.1, label='Americas Working Hours')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "15_timezone_inference.png"))
    plt.close()
    print(f"{Colors.GREEN}[DONE]{Colors.ENDC}")

    print(f"\n{Colors.GREEN}Analysis Complete. Visualizations saved to '{output_folder}/'{Colors.ENDC}")

# ==========================================
# MAIN
# ==========================================

def main():
    print_banner()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='mastodon_analyzed.json')
    parser.add_argument('--output-dir', type=str, default='figures')
    args = parser.parse_args()
    
    df = load_data(args.input)
    generate_visualizations(df, args.output_dir)

if __name__ == "__main__":
    main()