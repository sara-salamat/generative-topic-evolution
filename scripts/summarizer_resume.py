import os
import requests
from tqdm import tqdm
import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.classes.init import Auth
from dotenv import load_dotenv
import json
import time

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# --- CONFIG ---
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
CLASS_NAME = "Papers"
OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"
BATCH_SIZE = 50
PROGRESS_FILE = "summarizer_progress.json"

# --- CONNECT ---
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    skip_init_checks=True,
)
collection = client.collections.get(CLASS_NAME)

# --- Ensure 'summary' property exists ---
schema = collection.config.get()
if not any(p.name == "summary" for p in schema.properties):
    print("Adding 'summary' property to schema...")
    collection.config.add_property(Property(name="summary", data_type=DataType.TEXT))

# --- Helper: Summarize with Qwen via Ollama ---
def summarize_abstract(abstract, max_retries=3):
    prompt = f"""Create a concise academic summary of the following research abstract.

Focus on:
- Research question/objective
- Methodology/approach
- Key findings/contributions
- Impact/significance

Keep it to 2-3 sentences maximum.

Abstract: {abstract}

Summary:"""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(OLLAMA_API, json=payload, timeout=30)
            response.raise_for_status()
            summary = response.json()["response"].strip()
            
            # Remove thinking process if present
            if "<think>" in summary and "</think>" in summary:
                # Extract only the part after </think>
                summary = summary.split("</think>")[-1].strip()
            
            # Clean up any remaining artifacts
            summary = summary.replace("<think>", "").replace("</think>", "").strip()
            
            return summary
        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries - 1:
                print(f"Connection error (attempt {attempt + 1}/{max_retries}), retrying in 10 seconds...")
                time.sleep(10)
            else:
                print(f"Failed to connect to Ollama after {max_retries} attempts: {e}")
                return None  # Return None to indicate failure
        except requests.exceptions.Timeout as e:
            if attempt < max_retries - 1:
                print(f"Timeout error (attempt {attempt + 1}/{max_retries}), retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Timeout after {max_retries} attempts: {e}")
                return None
        except Exception as e:
            print(f"Error summarizing: {e}")
            return None

# --- Load progress ---
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"processed_uuids": [], "last_batch": 0}

def save_progress(processed_uuids, last_batch):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({"processed_uuids": processed_uuids, "last_batch": last_batch}, f)

# --- Main processing ---
print("Loading progress...")
progress = load_progress()
processed_uuids = set(progress["processed_uuids"])
last_batch = progress["last_batch"]

print(f"Already processed: {len(processed_uuids)} papers")
print(f"Last batch: {last_batch}")

# --- Fetch all objects ---
print("Fetching all records from Weaviate...")
objs = list(collection.iterator())
print(f"Found {len(objs)} records.")

# --- Filter objects that need processing ---
objects_to_process = []
for obj in objs:
    abstract = obj.properties.get("abstract", "")
    if not abstract:
        continue
    if obj.properties.get("summary"):
        continue
    if obj.uuid in processed_uuids:
        continue
    objects_to_process.append(obj)

print(f"Need to process {len(objects_to_process)} papers")

# --- Process in batches ---
processed_count = 0
error_count = 0
consecutive_failures = 0
max_consecutive_failures = 10

for i in tqdm(range(0, len(objects_to_process), BATCH_SIZE), desc="Processing batches"):
    batch = objects_to_process[i:i + BATCH_SIZE]
    batch_failures = 0
    
    # Process batch
    for obj in batch:
        summary = summarize_abstract(obj.properties.get("abstract", ""))
        
        if summary is None:  # Ollama failed
            batch_failures += 1
            consecutive_failures += 1
            error_count += 1
            continue
        
        if summary:  # Success
            try:
                collection.data.update(
                    uuid=obj.uuid,
                    properties={"summary": summary}
                )
                processed_count += 1
                processed_uuids.add(obj.uuid)
                consecutive_failures = 0  # Reset consecutive failures
            except Exception as e:
                print(f"Error updating object {obj.uuid}: {e}")
                error_count += 1
        else:
            error_count += 1
    
    # Save progress after each batch
    save_progress(list(processed_uuids), i // BATCH_SIZE)
    
    # Check if we should stop due to too many consecutive failures
    if consecutive_failures >= max_consecutive_failures:
        print(f"\nStopping due to {consecutive_failures} consecutive failures")
        print("Please check if Ollama is running and restart the script")
        break
    
    # Small delay between batches
    time.sleep(1)

print(f"\nSummary: Processed {processed_count} papers, Errors: {error_count}")
print(f"Total processed so far: {len(processed_uuids)}")
print("Summarization and update complete.")
client.close() 