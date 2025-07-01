import os
import json
from tqdm import tqdm
from uuid import UUID

import weaviate
from weaviate.classes.data import DataObject
from weaviate.classes.config import Property, DataType, Configure, VectorDistances
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter

# --- 🔧 CONFIG ---
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]          
INPUT_PATH = "/mnt/data/sara-salamat/generative-topic-evolution/data/processed/embedded_records.json"
CLASS_NAME = "Papers"
BATCH_SIZE = 100

# --- CONNECT ---
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    skip_init_checks=True,  # <--- disables gRPC ping on startup
    
)

if not client.is_ready():
    raise RuntimeError("Weaviate cluster is not ready")

# --- CLEAR EXISTING DATA ---
print(f"Clearing all previous records from '{CLASS_NAME}'...")
client.collections.delete(CLASS_NAME)

# --- SCHEMA ---
if not client.collections.exists(CLASS_NAME):
    client.collections.create(
        name=CLASS_NAME,
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="abstract", data_type=DataType.TEXT),
            Property(name="conference", data_type=DataType.TEXT),
            Property(name="venue", data_type=DataType.TEXT),
            Property(name="keywords", data_type=DataType.TEXT_ARRAY),
            Property(name="paper_id", data_type=DataType.TEXT),
            Property(name="year", data_type=DataType.INT),
        ],
        vector_index_config=Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE)
    )

collection = client.collections.get(CLASS_NAME)

# --- LOAD & UPLOAD ---
with open(INPUT_PATH, "r") as f:
    records = json.load(f)

for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Uploading"):
    batch = records[i : i + BATCH_SIZE]
    objs = []

    for r in batch:
        venue = r.get("venue")
        if isinstance(venue, dict):
            venue = venue.get("value", "")
        venue = venue if isinstance(venue, str) else "unknown"

        # Clean keywords
        raw_keywords = r.get("keywords", [])
        if isinstance(raw_keywords, dict) and isinstance(raw_keywords.get("value"), list):
            raw_keywords = raw_keywords["value"]

        keywords = [kw for kw in raw_keywords if isinstance(kw, str)]

        objs.append(DataObject(
            properties={
                "title": r["title"],
                "abstract": r["abstract"],
                "conference": r["conference"],
                "venue": venue,
                "paper_id": r["id"],
                "year": int(r["conference"][-4:]),
                "keywords": keywords
            },
            vector=r["embedding"],
        ))

    collection.data.insert_many(objs)

print("Upload complete.")
client.close()