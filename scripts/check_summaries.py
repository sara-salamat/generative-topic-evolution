import os
import weaviate
from weaviate.classes.init import Auth
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# --- CONFIG ---
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
CLASS_NAME = "Papers"

# --- CONNECT ---
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    skip_init_checks=True,
)
collection = client.collections.get(CLASS_NAME)

# --- Check summaries ---
print("Checking existing summaries...")

total_count = 0
with_summary_count = 0
without_summary_count = 0
without_abstract_count = 0

# Sample a few objects to check
sample_objects = []
for obj in collection.iterator():
    total_count += 1
    
    if total_count <= 5:  # Show first 5 objects for debugging
        sample_objects.append(obj)
    
    abstract = obj.properties.get("abstract", "")
    summary = obj.properties.get("summary", "")
    
    if not abstract:
        without_abstract_count += 1
    elif summary:
        with_summary_count += 1
    else:
        without_summary_count += 1
    
    if total_count % 1000 == 0:
        print(f"Processed {total_count} objects...")

print(f"\n=== SUMMARY ===")
print(f"Total papers: {total_count}")
print(f"Papers with summaries: {with_summary_count}")
print(f"Papers without summaries: {without_summary_count}")
print(f"Papers without abstracts: {without_abstract_count}")

print(f"\n=== SAMPLE OBJECTS ===")
for i, obj in enumerate(sample_objects):
    print(f"\nObject {i+1}:")
    print(f"  UUID: {obj.uuid}")
    print(f"  Title: {obj.properties.get('title', 'N/A')[:100]}...")
    print(f"  Has abstract: {'Yes' if obj.properties.get('abstract') else 'No'}")
    print(f"  Has summary: {'Yes' if obj.properties.get('summary') else 'No'}")
    if obj.properties.get('summary'):
        print(f"  Summary: {obj.properties.get('summary')[:200]}...")

client.close() 