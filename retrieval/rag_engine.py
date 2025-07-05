import logging
import requests
import json
from weaviate_retriever import WeaviateRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRAG:
    def __init__(self, retriever, api_base="http://localhost:11434", model="qwen2.5:32b"):
        self.retriever = retriever
        self.api_base = api_base
        self.model = model

    def query_llm(self, prompt):
        """Query the Ollama LLM directly via API"""
        url = f"{self.api_base}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return f"Error: {str(e)}"

    def answer(self, question):
        """Retrieve context and generate answer"""
        # Retrieve relevant context
        context_chunks = self.retriever.retrieve(question)
        context = " ".join(context_chunks)
        
        # Create prompt
        prompt = f"""Based on the following context, answer the question.

            Context: {context}

            Question: {question}

            Answer:"""
                
        # Get answer from LLM
        answer = self.query_llm(prompt)
        return answer

    def build_context(self, context_chunks):
        pass  # TODO: add like year: paper: [title,abstract]

if __name__ == "__main__":
    retriever = WeaviateRetriever()
    rag = SimpleRAG(retriever=retriever)

    # Test the RAG system
    question = "What are transformer-based methods for recommendation in 2022?"
    answer = rag.answer(question)
    print("QUESTION:", question)
    print("ANSWER:", answer)

    retriever.close()

