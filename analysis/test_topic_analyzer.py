#!/usr/bin/env python3
"""
Test script for the TopicEvolutionAnalyzer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from analysis.topic_evolution_analyzer import TopicEvolutionAnalyzer
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

def test_basic_functionality():
    """Test basic functionality of the analyzer"""
    print("=== Testing TopicEvolutionAnalyzer ===")
    
    analyzer = TopicEvolutionAnalyzer()
    
    try:
        # Test 1: Get papers by year
        print("\n1. Testing paper retrieval...")
        papers_2023 = analyzer.get_papers_by_year(2023, limit=10)
        print(f"   Retrieved {len(papers_2023)} papers from 2023")
        
        if papers_2023:
            print(f"   Sample paper: {papers_2023[0]['title'][:50]}...")
        
        # Test 2: Extract topics
        print("\n2. Testing topic extraction...")
        if papers_2023:
            texts = [p.get('title', '') + " " + p.get('abstract', '') + " " + p.get('summary', '') 
                    for p in papers_2023[:5]]
            topics = analyzer.extract_topics_from_text(texts, n_topics=5)
            print(f"   Extracted topics: {topics}")
        
        # Test 3: Track trends for specific topics
        print("\n3. Testing trend tracking...")
        key_topics = ["transformer", "recommendation", "machine learning"]
        trends = analyzer.track_topic_trends(key_topics, (2022, 2024))
        print(f"   Tracked trends for {len(trends)} topics")
        
        # Test 4: Identify emerging topics
        print("\n4. Testing emerging topic identification...")
        emerging = analyzer.identify_emerging_topics([2023, 2024], [2020, 2021])
        if 'emerging_topics' in emerging:
            print(f"   Found {len(emerging['emerging_topics'])} emerging topics")
            for topic, data in list(emerging['emerging_topics'].items())[:3]:
                print(f"     - {topic}: {data['growth_rate']:.2%} growth")
        
        # Test 5: Generate comprehensive report
        print("\n5. Testing comprehensive report generation...")
        report = analyzer.generate_trend_report((2022, 2024))
        print(f"   Generated report with {report.get('total_papers', 0)} papers")
        
        # Save test report
        analyzer.save_report(report, "test_topic_evolution_report.json")
        print("   Test report saved to test_topic_evolution_report.json")
        
        return True
        
    except Exception as e:
        print(f"   Error during testing: {e}")
        return False
    finally:
        analyzer.close()

def create_visualizations():
    """Create sample visualizations from the analyzer output"""
    print("\n=== Creating Visualizations ===")
    
    try:
        # Load the test report
        with open("test_topic_evolution_report.json", 'r') as f:
            report = json.load(f)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Topic Evolution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Papers per year
        ax1 = axes[0, 0]
        years = list(report['papers_per_year'].keys())
        counts = list(report['papers_per_year'].values())
        ax1.bar(years, counts, color='skyblue', alpha=0.7)
        ax1.set_title('Papers per Year')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of Papers')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Topic trends (if available)
        ax2 = axes[0, 1]
        if 'topic_trends' in report:
            trends = report['topic_trends']
            for topic, trend_data in list(trends.items())[:5]:  # Top 5 topics
                years = [item['year'] for item in trend_data]
                percentages = [item['percentage'] for item in trend_data]
                ax2.plot(years, percentages, marker='o', label=topic, linewidth=2)
            ax2.set_title('Topic Trends Over Time')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Percentage of Papers')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Emerging topics
        ax3 = axes[1, 0]
        if 'emerging_topics' in report and 'emerging_topics' in report['emerging_topics']:
            emerging = report['emerging_topics']['emerging_topics']
            top_emerging = list(emerging.items())[:10]  # Top 10 emerging topics
            
            topics = [item[0] for item in top_emerging]
            growth_rates = [item[1]['growth_rate'] for item in top_emerging]
            
            colors = ['red' if item[1]['is_new'] else 'orange' for item in top_emerging]
            
            bars = ax3.barh(topics, growth_rates, color=colors, alpha=0.7)
            ax3.set_title('Top Emerging Topics')
            ax3.set_xlabel('Growth Rate')
            ax3.set_ylabel('Topic')
            
            # Add value labels on bars
            for i, (bar, rate) in enumerate(zip(bars, growth_rates)):
                ax3.text(rate + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{rate:.1%}', va='center', fontsize=8)
        
        # 4. Topic relationships (centrality)
        ax4 = axes[1, 1]
        if 'topic_relationships' in report and 'centrality' in report['topic_relationships']:
            centrality = report['topic_relationships']['centrality']
            top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            
            topics = [item[0] for item in top_central]
            centrality_scores = [item[1] for item in top_central]
            
            ax4.barh(topics, centrality_scores, color='green', alpha=0.7)
            ax4.set_title('Topic Centrality (Most Connected)')
            ax4.set_xlabel('Number of Connections')
            ax4.set_ylabel('Topic')
        
        plt.tight_layout()
        plt.savefig('topic_evolution_visualizations.png', dpi=300, bbox_inches='tight')
        print("   Visualizations saved to topic_evolution_visualizations.png")
        
        # Create a summary text file
        with open('topic_evolution_summary.txt', 'w') as f:
            f.write("TOPIC EVOLUTION ANALYSIS SUMMARY\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Year Range: {report['year_range']}\n")
            f.write(f"Total Papers: {report['total_papers']}\n\n")
            
            f.write("PAPERS PER YEAR:\n")
            for year, count in report['papers_per_year'].items():
                f.write(f"  {year}: {count} papers\n")
            
            f.write(f"\nKEY TOPICS IDENTIFIED:\n")
            for i, topic in enumerate(report['key_topics'][:10], 1):
                f.write(f"  {i}. {topic}\n")
            
            if 'emerging_topics' in report and 'emerging_topics' in report['emerging_topics']:
                f.write(f"\nTOP EMERGING TOPICS:\n")
                emerging = report['emerging_topics']['emerging_topics']
                for i, (topic, data) in enumerate(list(emerging.items())[:10], 1):
                    status = "NEW" if data['is_new'] else "GROWING"
                    f.write(f"  {i}. {topic} ({data['growth_rate']:.1%} growth) [{status}]\n")
        
        print("   Summary saved to topic_evolution_summary.txt")
        
    except Exception as e:
        print(f"   Error creating visualizations: {e}")

def main():
    """Main test function"""
    print("Starting TopicEvolutionAnalyzer tests...")
    
    # Test basic functionality
    success = test_basic_functionality()
    
    if success:
        # Create visualizations
        create_visualizations()
        print("\n✅ All tests completed successfully!")
        print("\nGenerated files:")
        print("  - test_topic_evolution_report.json (detailed report)")
        print("  - topic_evolution_visualizations.png (charts)")
        print("  - topic_evolution_summary.txt (text summary)")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 