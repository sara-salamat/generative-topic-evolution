# retriever.py
import os
import logging
import torch
from transformers import AutoTokenizer, AutoModel
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Config ---
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
CLASS_NAME = "Papers"

logger.info(f"Connecting to Weaviate at {WEAVIATE_URL}")
logger.info(f"Using collection: {CLASS_NAME}")

# --- Connect to Weaviate ---
try:
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        skip_init_checks=True,
    )
    collection = client.collections.get(CLASS_NAME)
    logger.info("Successfully connected to Weaviate and retrieved collection")
except Exception as e:
    logger.error(f"Failed to connect to Weaviate: {e}")
    raise

# Load model once
logger.info("Loading SPECTER model and tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
    model = AutoModel.from_pretrained("allenai/specter")
    logger.info("Successfully loaded SPECTER model and tokenizer")
except Exception as e:
    logger.error(f"Failed to load SPECTER model: {e}")
    raise

def get_specter_embedding(text: str):
    logger.debug(f"Generating embedding for text: {text[:100]}...")
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            output = model(**inputs)
        embedding = output.last_hidden_state.mean(dim=1)[0].numpy()
        logger.debug(f"Generated embedding with shape: {embedding.shape}")
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise

def retrieve_similar_papers(query: str, year: Optional[int] = None, limit: int = 5):
    logger.info(f"Retrieving similar papers for query: '{query[:100]}...'")
    logger.info(f"Parameters: year={year}, limit={limit}")
    
    try:
        vector = get_specter_embedding(query)
        logger.debug(f"Generated query vector with shape: {vector.shape}")
        
        filters = Filter.by_property("year").equal(year) if year else None
        if filters:
            logger.info(f"Applied year filter: {year}")
        else:
            logger.info("No year filter applied")
        
        logger.debug("Executing vector similarity search...")
        results = collection.query.near_vector(
            near_vector=vector,
            limit=limit,
            filters=filters
        )
        
        papers = [obj.properties for obj in results.objects]
        logger.info(f"Retrieved {len(papers)} papers")
        
        # Log some details about retrieved papers
        for i, paper in enumerate(papers):
            title = str(paper.get('title', 'No title'))
            year_value = paper.get('year', 'No year')
            logger.debug(f"Paper {i+1}: {title[:50]}... ({year_value})")
        
        return papers
        
    except Exception as e:
        logger.error(f"Failed to retrieve similar papers: {e}")
        raise

def retrieve_batch_queries(queries: list[str], year: Optional[int] = None, limit: int = 5):
    logger.info(f"Processing batch of {len(queries)} queries")
    logger.info(f"Batch parameters: year={year}, limit={limit}")
    
    results = {}
    for i, query in enumerate(queries):
        logger.info(f"Processing query {i+1}/{len(queries)}: '{query[:50]}...'")
        try:
            papers = retrieve_similar_papers(query, year, limit)
            results[query] = papers
            logger.info(f"Successfully processed query {i+1}, retrieved {len(papers)} papers")
        except Exception as e:
            logger.error(f"Failed to process query {i+1}: {e}")
            results[query] = []  # Return empty list for failed queries
    
    logger.info(f"Batch processing completed. Successfully processed {len([r for r in results.values() if r])} queries")
    return results

if __name__ == "__main__":
    queries = [
        "transformer models for recommendation",
        "knowledge graph",
    ]
    results = retrieve_batch_queries(queries, year=2024, limit=5)
    print(results)
    client.close()  # Close the Weaviate connection