import os
import requests
from tqdm import tqdm
import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.classes.init import Auth
from dotenv import load_dotenv
import json
import time
import subprocess
import psutil
import gc
from typing import Iterator, List, Dict, Any

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# --- CONFIG ---
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
CLASS_NAME = "Papers"
OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_HEALTH_API = "http://localhost:11434/api/tags"
OLLAMA_MODEL = "qwen3:8b"
BATCH_SIZE = 10  # Smaller batch size for memory efficiency
PROGRESS_FILE = "summarizer_progress.json"
MAX_CONSECUTIVE_FAILURES = 5
HEALTH_CHECK_INTERVAL = 5  # Check more frequently
MEMORY_THRESHOLD = 85  # Stop if memory usage exceeds 85%

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

def get_memory_usage() -> float:
    """Get current memory usage percentage"""
    return psutil.virtual_memory().percent

def check_memory_threshold() -> bool:
    """Check if memory usage is above threshold"""
    return get_memory_usage() > MEMORY_THRESHOLD

def force_garbage_collection():
    """Force garbage collection to free memory"""
    gc.collect()

# --- Ollama Health Check ---
def check_ollama_health():
    """Check if Ollama is running and the model is loaded"""
    try:
        # Check if Ollama service is responding
        response = requests.get(OLLAMA_HEALTH_API, timeout=5)
        if response.status_code != 200:
            return False, "Ollama service not responding"
        
        # Check if our model is loaded
        models = response.json().get("models", [])
        model_names = [model.get("name", "") for model in models]
        
        if OLLAMA_MODEL not in model_names:
            return False, f"Model {OLLAMA_MODEL} not loaded"
        
        return True, "OK"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama"
    except Exception as e:
        return False, f"Health check error: {e}"

def restart_ollama_if_needed():
    """Attempt to restart Ollama if it's not healthy"""
    print("Attempting to restart Ollama...")
    try:
        # Kill existing Ollama processes
        subprocess.run(["pkill", "-f", "ollama"], check=False)
        time.sleep(2)
        
        # Start Ollama
        subprocess.Popen(["ollama", "serve"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        time.sleep(5)
        
        # Load the model
        subprocess.run(["ollama", "pull", OLLAMA_MODEL], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL)
        time.sleep(10)
        
        # Check if it's working now
        is_healthy, message = check_ollama_health()
        if is_healthy:
            print("Ollama restarted successfully")
            return True
        else:
            print(f"Ollama restart failed: {message}")
            return False
    except Exception as e:
        print(f"Failed to restart Ollama: {e}")
        return False

# --- Memory-efficient object fetching ---
def fetch_objects_in_batches(collection, batch_size: int = 100) -> Iterator[List[Dict[str, Any]]]:
    """Fetch objects from Weaviate in small batches to reduce memory usage"""
    try:
        # Get all objects and process them in batches
        all_objects = list(collection.iterator())
        print(f"Total objects fetched: {len(all_objects)}")
        
        # Process in batches
        for i in range(0, len(all_objects), batch_size):
            batch = all_objects[i:i + batch_size]
            
            # Convert to minimal dictionaries to save memory
            minimal_batch = []
            for obj in batch:
                minimal_batch.append({
                    'uuid': obj.uuid,
                    'abstract': obj.properties.get("abstract", ""),
                    'has_summary': bool(obj.properties.get("summary"))
                })
            
            yield minimal_batch
            
            # Force garbage collection after each batch
            del batch
            force_garbage_collection()
            
    except Exception as e:
        print(f"Error fetching objects: {e}")
        yield []

# --- Helper: Summarize with Qwen via Ollama ---
def summarize_abstract(abstract: str, max_retries: int = 3) -> str | None:
    """Summarize abstract with memory-efficient error handling"""
    if not abstract.strip():
        return ""
    
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
            # Check memory before making request
            if check_memory_threshold():
                print(f"Memory usage high ({get_memory_usage():.1f}%), waiting...")
                time.sleep(30)
                force_garbage_collection()
                continue
            
            # Check Ollama health before each attempt
            is_healthy, message = check_ollama_health()
            if not is_healthy:
                print(f"Ollama health check failed: {message}")
                if attempt == 0:  # Only try restart on first attempt
                    if restart_ollama_if_needed():
                        continue  # Try again after restart
                time.sleep(10)
                continue
            
            response = requests.post(OLLAMA_API, json=payload, timeout=60)
            response.raise_for_status()
            summary = response.json()["response"].strip()
            
            # Remove thinking process if present
            if "<think>" in summary and "</think>" in summary:
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
                return None
        except requests.exceptions.Timeout as e:
            if attempt < max_retries - 1:
                print(f"Timeout error (attempt {attempt + 1}/{max_retries}), retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Timeout after {max_retries} attempts: {e}")
                return None
        except Exception as e:
            print(f"Error summarizing: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return None
    
    return None

# --- Load progress ---
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"processed_uuids": [], "last_batch": 0}

def save_progress(processed_uuids, last_batch):
    with open(PROGRESS_FILE, 'w') as f:
        # Convert UUIDs to strings for JSON serialization
        uuid_list = [str(uuid) for uuid in processed_uuids]
        json.dump({"processed_uuids": uuid_list, "last_batch": last_batch}, f)

# --- Main processing ---
def main():
    print("Loading progress...")
    progress = load_progress()
    processed_uuids = set(progress["processed_uuids"])
    last_batch = progress["last_batch"]

    print(f"Already processed: {len(processed_uuids)} papers")
    print(f"Last batch: {last_batch}")
    print(f"Current memory usage: {get_memory_usage():.1f}%")

    # Initial Ollama health check
    print("Checking Ollama health...")
    is_healthy, message = check_ollama_health()
    if not is_healthy:
        print(f"Initial Ollama health check failed: {message}")
        print("Attempting to restart Ollama...")
        if not restart_ollama_if_needed():
            print("Failed to start Ollama. Please start it manually and run the script again.")
            return
    else:
        print("Ollama is healthy")

    # Process objects in memory-efficient batches
    processed_count = 0
    error_count = 0
    consecutive_failures = 0
    batch_count = 0
    
    print("Starting memory-efficient processing...")
    
    for batch in fetch_objects_in_batches(collection, batch_size=50):
        batch_count += 1
        batch_failures = 0
        
        # Filter objects that need processing
        objects_to_process = []
        for obj in batch:
            if not obj['abstract']:
                continue
            if obj['has_summary']:
                continue
            if obj['uuid'] in processed_uuids:
                continue
            objects_to_process.append(obj)
        
        if not objects_to_process:
            continue
        
        print(f"\nProcessing batch {batch_count}: {len(objects_to_process)} papers")
        print(f"Memory usage: {get_memory_usage():.1f}%")
        
        # Health check every few batches
        if batch_count % HEALTH_CHECK_INTERVAL == 0:
            is_healthy, message = check_ollama_health()
            if not is_healthy:
                print(f"Ollama health check failed: {message}")
                if restart_ollama_if_needed():
                    consecutive_failures = 0
                else:
                    print("Failed to restart Ollama. Stopping.")
                    break
        
        # Process objects in smaller sub-batches
        for i in range(0, len(objects_to_process), BATCH_SIZE):
            sub_batch = objects_to_process[i:i + BATCH_SIZE]
            
            for obj in sub_batch:
                # Check memory threshold
                if check_memory_threshold():
                    print(f"Memory usage high ({get_memory_usage():.1f}%), pausing...")
                    time.sleep(60)
                    force_garbage_collection()
                
                summary = summarize_abstract(obj['abstract'])
                
                if summary is None:  # Ollama failed
                    batch_failures += 1
                    consecutive_failures += 1
                    error_count += 1
                    continue
                
                if summary:  # Success
                    try:
                        collection.data.update(
                            uuid=obj['uuid'],
                            properties={"summary": summary}
                        )
                        processed_count += 1
                        processed_uuids.add(obj['uuid'])
                        consecutive_failures = 0
                    except Exception as e:
                        print(f"Error updating object {obj['uuid']}: {e}")
                        error_count += 1
                else:
                    error_count += 1
            
            # Save progress after each sub-batch
            save_progress(list(processed_uuids), batch_count)
            
            # Check if we should stop due to too many consecutive failures
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\nStopping due to {consecutive_failures} consecutive failures")
                print("Please check if Ollama is running and restart the script")
                return
            
            # Small delay between sub-batches
            time.sleep(1)
        
        # Force garbage collection after each batch
        force_garbage_collection()
        
        # Longer delay between batches
        time.sleep(3)
    
    print(f"\nSummary: Processed {processed_count} papers, Errors: {error_count}")
    print(f"Total processed so far: {len(processed_uuids)}")
    print(f"Final memory usage: {get_memory_usage():.1f}%")
    print("Summarization and update complete.")

if __name__ == "__main__":
    try:
        main()
    finally:
        client.close()
        force_garbage_collection() 