import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from datetime import datetime
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
from dotenv import load_dotenv
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopicEvolutionAnalyzer:
    """
    Analyzes the evolution of research topics over time in academic conferences.
    """
    
    def __init__(self, ollama_api="http://localhost:11434/api/generate", model="qwen2.5:32b"):
        self.weaviate_url = os.environ["WEAVIATE_URL"]
        self.api_key = os.environ["WEAVIATE_API_KEY"]
        self.ollama_api = ollama_api
        self.model = model
        
        # Connect to Weaviate
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.weaviate_url,
            auth_credentials=Auth.api_key(self.api_key),
            skip_init_checks=True,
        )
        self.collection = self.client.collections.get("Papers")
        
        # Initialize TF-IDF vectorizer for topic extraction
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
    def get_papers_by_year(self, year: int, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetch papers for a specific year"""
        try:
            filters = Filter.by_property("year").equal(year)
            results = self.collection.query.fetch_objects(
                filters=filters,
                limit=limit,
                include_vector=False
            )
            
            papers = []
            for obj in results.objects:
                papers.append({
                    'uuid': obj.uuid,
                    'title': obj.properties.get('title', ''),
                    'abstract': obj.properties.get('abstract', ''),
                    'summary': obj.properties.get('summary', ''),
                    'year': obj.properties.get('year', year),
                    'venue': obj.properties.get('venue', '')
                })
            
            logger.info(f"Retrieved {len(papers)} papers for year {year}")
            return papers
            
        except Exception as e:
            logger.error(f"Error fetching papers for year {year}: {e}")
            return []
    
    def extract_topics_from_text(self, texts: List[str], n_topics: int = 10) -> List[str]:
        """Extract main topics from a list of texts using TF-IDF and clustering"""
        if not texts:
            return []
        
        # Combine all texts
        combined_text = " ".join(texts)
        
        # Use LLM to extract key topics
        prompt = f"""Extract the top {n_topics} most important research topics from the following academic texts.
        
        Focus on:
        - Technical methods and approaches
        - Research domains and fields
        - Emerging technologies
        - Application areas
        
        Return only the topic names, one per line, without numbering or explanations.
        
        Texts: {combined_text[:4000]}...
        
        Topics:"""
        
        try:
            response = requests.post(
                self.ollama_api,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            
            topics = response.json()["response"].strip().split('\n')
            # Clean up topics
            topics = [topic.strip().replace('- ', '').replace('* ', '') for topic in topics if topic.strip()]
            return topics[:n_topics]
            
        except Exception as e:
            logger.error(f"Error extracting topics with LLM: {e}")
            # Fallback to TF-IDF
            return self._extract_topics_tfidf(texts, n_topics)
    
    def _extract_topics_tfidf(self, texts: List[str], n_topics: int = 10) -> List[str]:
        """Fallback topic extraction using TF-IDF"""
        try:
            # Fit TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get top terms
            tfidf_sums = np.array(tfidf_matrix.sum(axis=0)).flatten()
            top_indices = tfidf_sums.argsort()[-n_topics:][::-1]
            
            return [feature_names[i] for i in top_indices]
        except Exception as e:
            logger.error(f"Error in TF-IDF topic extraction: {e}")
            return []
    
    def track_topic_trends(self, topic_keywords: List[str], year_range: Tuple[int, int]) -> Dict[str, Any]:
        """
        Track how specific topics evolve over years.
        
        Args:
            topic_keywords: List of topic keywords to track
            year_range: Tuple of (start_year, end_year)
        
        Returns:
            Dictionary with topic trend data
        """
        start_year, end_year = year_range
        trends = {topic: [] for topic in topic_keywords}
        
        for year in range(start_year, end_year + 1):
            papers = self.get_papers_by_year(year)
            
            if not papers:
                continue
            
            # Combine all text for the year
            year_text = " ".join([
                paper.get('title', '') + " " + paper.get('abstract', '') + " " + paper.get('summary', '')
                for paper in papers
            ]).lower()
            
            # Count topic mentions
            for topic in topic_keywords:
                topic_lower = topic.lower()
                count = year_text.count(topic_lower)
                trends[topic].append({
                    'year': year,
                    'count': count,
                    'percentage': count / len(papers) * 100 if papers else 0
                })
        
        return trends
    
    def identify_emerging_topics(self, recent_years: List[int], comparison_years: List[int], 
                               threshold: float = 0.1) -> Dict[str, Any]:
        """
        Identify newly emerging research topics by comparing recent years to earlier years.
        
        Args:
            recent_years: List of recent years to analyze
            comparison_years: List of earlier years for comparison
            threshold: Minimum growth threshold to consider a topic "emerging"
        
        Returns:
            Dictionary with emerging topics and their growth metrics
        """
        # Get papers for recent and comparison periods
        recent_papers = []
        comparison_papers = []
        
        for year in recent_years:
            recent_papers.extend(self.get_papers_by_year(year))
        
        for year in comparison_years:
            comparison_papers.extend(self.get_papers_by_year(year))
        
        # Extract topics from both periods
        recent_texts = [p.get('title', '') + " " + p.get('abstract', '') + " " + p.get('summary', '') 
                       for p in recent_papers]
        comparison_texts = [p.get('title', '') + " " + p.get('abstract', '') + " " + p.get('summary', '') 
                           for p in comparison_papers]
        
        recent_topics = self.extract_topics_from_text(recent_texts, n_topics=20)
        comparison_topics = self.extract_topics_from_text(comparison_texts, n_topics=20)
        
        # Calculate topic frequencies
        recent_freq = Counter()
        comparison_freq = Counter()
        
        for text in recent_texts:
            text_lower = text.lower()
            for topic in recent_topics:
                recent_freq[topic] += text_lower.count(topic.lower())
        
        for text in comparison_texts:
            text_lower = text.lower()
            for topic in comparison_topics:
                comparison_freq[topic] += text_lower.count(topic.lower())
        
        # Normalize frequencies by number of papers
        recent_total = len(recent_papers)
        comparison_total = len(comparison_papers)
        
        emerging_topics = {}
        for topic in recent_topics:
            recent_rate = recent_freq[topic] / recent_total if recent_total > 0 else 0
            comparison_rate = comparison_freq[topic] / comparison_total if comparison_total > 0 else 0
            
            if comparison_rate > 0:
                growth_rate = (recent_rate - comparison_rate) / comparison_rate
            else:
                growth_rate = recent_rate  # New topic
            
            if growth_rate >= threshold:
                emerging_topics[topic] = {
                    'recent_frequency': recent_rate,
                    'comparison_frequency': comparison_rate,
                    'growth_rate': growth_rate,
                    'is_new': comparison_rate == 0
                }
        
        # Sort by growth rate
        emerging_topics = dict(sorted(emerging_topics.items(), 
                                     key=lambda x: x[1]['growth_rate'], reverse=True))
        
        return {
            'emerging_topics': emerging_topics,
            'recent_years': recent_years,
            'comparison_years': comparison_years,
            'total_recent_papers': recent_total,
            'total_comparison_papers': comparison_total
        }
    
    def analyze_topic_relationships(self, year: int, min_cooccurrence: int = 2) -> Dict[str, Any]:
        """
        Analyze topic co-occurrence and relationships within a specific year.
        
        Args:
            year: Year to analyze
            min_cooccurrence: Minimum number of co-occurrences to include
        
        Returns:
            Dictionary with topic relationships and network data
        """
        papers = self.get_papers_by_year(year)
        
        if not papers:
            return {'error': f'No papers found for year {year}'}
        
        # Extract topics from all papers
        all_texts = [p.get('title', '') + " " + p.get('abstract', '') + " " + p.get('summary', '') 
                    for p in papers]
        topics = self.extract_topics_from_text(all_texts, n_topics=30)
        
        # Create topic-paper matrix
        topic_paper_matrix = {}
        for i, paper in enumerate(papers):
            paper_text = (paper.get('title', '') + " " + paper.get('abstract', '') + 
                         " " + paper.get('summary', '')).lower()
            
            for topic in topics:
                if topic.lower() in paper_text:
                    if topic not in topic_paper_matrix:
                        topic_paper_matrix[topic] = set()
                    topic_paper_matrix[topic].add(i)
        
        # Calculate co-occurrence matrix
        cooccurrence_matrix = {}
        topic_frequencies = {}
        
        for topic1 in topics:
            topic_frequencies[topic1] = len(topic_paper_matrix.get(topic1, set()))
            cooccurrence_matrix[topic1] = {}
            
            for topic2 in topics:
                if topic1 != topic2:
                    papers_with_topic1 = topic_paper_matrix.get(topic1, set())
                    papers_with_topic2 = topic_paper_matrix.get(topic2, set())
                    cooccurrence = len(papers_with_topic1.intersection(papers_with_topic2))
                    
                    if cooccurrence >= min_cooccurrence:
                        cooccurrence_matrix[topic1][topic2] = cooccurrence
        
        # Calculate topic centrality (simple degree centrality)
        centrality = {}
        for topic in topics:
            centrality[topic] = len([t for t in cooccurrence_matrix[topic].values() if t > 0])
        
        return {
            'year': year,
            'topics': topics,
            'topic_frequencies': topic_frequencies,
            'cooccurrence_matrix': cooccurrence_matrix,
            'centrality': centrality,
            'total_papers': len(papers)
        }
    
    def generate_trend_report(self, year_range: Tuple[int, int], 
                            key_topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive trend report for the specified year range.
        
        Args:
            year_range: Tuple of (start_year, end_year)
            key_topics: Optional list of specific topics to track
        
        Returns:
            Comprehensive trend report
        """
        start_year, end_year = year_range
        
        # Get all papers in the range
        all_papers = []
        for year in range(start_year, end_year + 1):
            all_papers.extend(self.get_papers_by_year(year))
        
        if not all_papers:
            return {'error': f'No papers found for year range {year_range}'}
        
        # Extract key topics if not provided
        if not key_topics:
            all_texts = [p.get('title', '') + " " + p.get('abstract', '') + " " + p.get('summary', '') 
                        for p in all_papers]
            key_topics = self.extract_topics_from_text(all_texts, n_topics=15)
        
        # Track trends for key topics
        trends = self.track_topic_trends(key_topics, year_range)
        
        # Identify emerging topics (compare last 2 years to previous 2 years)
        recent_years = [end_year - 1, end_year]
        comparison_years = [start_year, start_year + 1]
        emerging = self.identify_emerging_topics(recent_years, comparison_years)
        
        # Analyze relationships for the most recent year
        relationships = self.analyze_topic_relationships(end_year)
        
        # Calculate overall statistics
        total_papers = len(all_papers)
        papers_per_year = {}
        for year in range(start_year, end_year + 1):
            papers_per_year[year] = len([p for p in all_papers if p.get('year') == year])
        
        return {
            'year_range': year_range,
            'total_papers': total_papers,
            'papers_per_year': papers_per_year,
            'key_topics': key_topics,
            'topic_trends': trends,
            'emerging_topics': emerging,
            'topic_relationships': relationships,
            'generated_at': datetime.now().isoformat()
        }
    
    def save_report(self, report: Dict[str, Any], filename: str):
        """Save a trend report to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
    def close(self):
        """Close Weaviate connection"""
        self.client.close()


def main():
    """Example usage of the TopicEvolutionAnalyzer"""
    analyzer = TopicEvolutionAnalyzer()
    
    try:
        # Example: Analyze trends from 2020 to 2024
        print("Generating comprehensive trend report...")
        report = analyzer.generate_trend_report((2020, 2024))
        
        # Save the report
        analyzer.save_report(report, "topic_evolution_report_2020_2024.json")
        
        # Print summary
        print(f"\n=== TOPIC EVOLUTION REPORT ===")
        print(f"Year Range: {report['year_range']}")
        print(f"Total Papers: {report['total_papers']}")
        print(f"Key Topics: {', '.join(report['key_topics'][:5])}...")
        
        if 'emerging_topics' in report and 'emerging_topics' in report['emerging_topics']:
            emerging = report['emerging_topics']['emerging_topics']
            print(f"\nTop Emerging Topics:")
            for i, (topic, data) in enumerate(list(emerging.items())[:5]):
                print(f"  {i+1}. {topic} (growth: {data['growth_rate']:.2%})")
        
        print(f"\nReport saved to: topic_evolution_report_2020_2024.json")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        analyzer.close()


if __name__ == "__main__":
    main() 