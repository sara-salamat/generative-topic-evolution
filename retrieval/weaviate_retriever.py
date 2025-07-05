import os
import logging
import sys
from typing import Optional, List
import torch
from transformers import AutoTokenizer, AutoModel
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.query_parser import parse_query 

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeaviateRetriever:
    def __init__(self, class_name="Papers"):
        self.weaviate_url = os.environ["WEAVIATE_URL"]
        self.api_key = os.environ["WEAVIATE_API_KEY"]
        self.class_name = class_name

        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.weaviate_url,
            auth_credentials=Auth.api_key(self.api_key),
            skip_init_checks=True,
        )
        self.collection = self.client.collections.get(self.class_name)

        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
        self.model = AutoModel.from_pretrained("allenai/specter")

    def embed(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            output = self.model(**inputs)
        return output.last_hidden_state.mean(dim=1)[0].numpy()

    def retrieve(self, query: str, year: Optional[int] = None, limit: int = 5) -> List[str]:
        logger.info(f"Raw query: {query}")
        parsed_query = parse_query(query) 
        logger.info(f"Parsed query: {parsed_query}")

        # Use the raw_query for embedding, or fall back to the original query
        query_text = parsed_query.get("raw_query", query)
        query_topic = parsed_query.get("topic_hint", None)
        if query_topic:
            vector = self.embed(query_topic)
        else:
            vector = self.embed(query_text)
        
        # Use year from parsed_query if available, otherwise use the year parameter
        year_to_filter = None
        if parsed_query.get("year_filter") and parsed_query["year_filter"].get("exact"):
            year_to_filter = parsed_query["year_filter"]["exact"]
        elif year:
            year_to_filter = year
            
        filters = Filter.by_property("year").equal(year_to_filter) if year_to_filter else None
        results = self.collection.query.near_vector(near_vector=vector, limit=limit, filters=filters)
        return [str(obj.properties["abstract"]) for obj in results.objects]

    def close(self):
        self.client.close()
