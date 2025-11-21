import json
import os
from elasticsearch import Elasticsearch, helpers

# ==========================================
# CONFIGURATION
# ==========================================

ES_HOST = "http://localhost:9200"
ES_INDEX = "mastodon_ai_events"
BACKUP_FILE = "mastodon_data_backup.json"

# ==========================================
# EXPORT FUNCTION (For You)
# ==========================================

def export_data():
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        print("❌ Elasticsearch is not running.")
        return

    print(f"⏳ Exporting data from index '{ES_INDEX}'...")
    
    try:
        # "scan" fetches all data efficiently
        query = {"query": {"match_all": {}}}
        scan_gen = helpers.scan(es, query=query, index=ES_INDEX)
        
        documents = []
        for doc in scan_gen:
            # We only save the source data, not the internal ES metadata
            documents.append(doc['_source'])
            
        with open(BACKUP_FILE, 'w', encoding='utf-8') as f:
            json.dump(documents, f, default=str) # default=str handles datetime objects
            
        print(f"✅ Successfully exported {len(documents)} records to '{BACKUP_FILE}'")
        print("👉 You can now commit this file to GitHub!")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")

# ==========================================
# IMPORT FUNCTION (For Your Team)
# ==========================================

def import_data():
    if not os.path.exists(BACKUP_FILE):
        print(f"❌ Backup file '{BACKUP_FILE}' not found.")
        return

    es = Elasticsearch(ES_HOST)
    if not es.ping():
        print("❌ Elasticsearch is not running. Start Docker first!")
        return

    print(f"⏳ Importing data from '{BACKUP_FILE}' into '{ES_INDEX}'...")
    
    try:
        with open(BACKUP_FILE, 'r', encoding='utf-8') as f:
            documents = json.load(f)
            
        # Prepare bulk actions
        actions = []
        for doc in documents:
            # Re-create the unique ID to prevent duplicates
            # We assume the doc has 'instance' and 'content' or similar unique fields
            # Or we just let ES generate a new ID if we don't care about overwriting
            
            action = {
                "_index": ES_INDEX,
                "_source": doc
            }
            actions.append(action)
            
        # Bulk insert
        helpers.bulk(es, actions)
        print(f"✅ Successfully imported {len(actions)} records.")
        print("🎉 You can now run 'analyze_from_es.py'")
        
    except Exception as e:
        print(f"❌ Import failed: {e}")

# ==========================================
# MENU
# ==========================================

if __name__ == "__main__":
    import sys
    
    print("--- Mastodon Data Manager ---")
    print("1. Export Data (Create Backup)")
    print("2. Import Data (Load Backup)")
    
    choice = input("Select option (1/2): ").strip()
    
    if choice == '1':
        export_data()
    elif choice == '2':
        import_data()
    else:
        print("Invalid choice.")