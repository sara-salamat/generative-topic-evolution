import os
from transformers import AutoTokenizer, AutoModel
import torch
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter

# --- Config ---
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
CLASS_NAME = "Papers"

# --- Connect ---
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    skip_init_checks=True,
)
collection = client.collections.get(CLASS_NAME)

# --- Embed query ---
query = "transformer models for recommendation"
tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
model = AutoModel.from_pretrained("allenai/specter")
def get_specter_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**inputs)
    embeddings = output.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings[0].numpy()  # Return as numpy vector

# --- Run query ---
results = collection.query.near_vector(
    near_vector=get_specter_embedding(query),
    limit=5,
)

# --- Print results ---
for obj in results.objects:
    print(f"- {obj.properties['title']} ({obj.properties.get('year')})")

client.close()
