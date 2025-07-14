#!/usr/bin/env python3
"""
Command-line interface for Topic Evolution Analysis
"""

import argparse
import sys
import os
from datetime import datetime
from analysis.topic_evolution_analyzer import TopicEvolutionAnalyzer

def main():
    parser = argparse.ArgumentParser(
        description="Analyze topic evolution in academic conferences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a comprehensive report for 2020-2024
  python run_topic_analysis.py --start-year 2020 --end-year 2024 --output report_2020_2024.json

  # Track specific topics
  python run_topic_analysis.py --start-year 2022 --end-year 2024 --topics "transformer,recommendation,machine learning"

  # Identify emerging topics (compare 2023-2024 vs 2020-2021)
  python run_topic_analysis.py --emerging --recent-years 2023,2024 --comparison-years 2020,2021

  # Quick test with limited data
  python run_topic_analysis.py --start-year 2023 --end-year 2024 --test-mode
        """
    )
    
    # Main analysis options
    parser.add_argument("--start-year", type=int, help="Start year for analysis")
    parser.add_argument("--end-year", type=int, help="End year for analysis")
    parser.add_argument("--output", default="topic_evolution_report.json", 
                       help="Output file for the report (default: topic_evolution_report.json)")
    
    # Topic tracking
    parser.add_argument("--topics", help="Comma-separated list of specific topics to track")
    
    # Emerging topics analysis
    parser.add_argument("--emerging", action="store_true", 
                       help="Focus on emerging topics analysis")
    parser.add_argument("--recent-years", help="Comma-separated recent years for emerging analysis")
    parser.add_argument("--comparison-years", help="Comma-separated comparison years for emerging analysis")
    parser.add_argument("--threshold", type=float, default=0.1,
                       help="Growth threshold for emerging topics (default: 0.1)")
    
    # Topic relationships
    parser.add_argument("--relationships", type=int, help="Analyze topic relationships for specific year")
    parser.add_argument("--min-cooccurrence", type=int, default=2,
                       help="Minimum co-occurrence for relationships (default: 2)")
    
    # Test mode
    parser.add_argument("--test-mode", action="store_true", 
                       help="Run in test mode with limited data")
    
    # Verbosity
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.start_year, args.emerging, args.relationships]):
        parser.error("Must specify either --start-year, --emerging, or --relationships")
    
    if args.start_year and not args.end_year:
        parser.error("--start-year requires --end-year")
    
    if args.emerging and (not args.recent_years or not args.comparison_years):
        parser.error("--emerging requires both --recent-years and --comparison-years")
    
    # Initialize analyzer
    print("Initializing Topic Evolution Analyzer...")
    analyzer = TopicEvolutionAnalyzer()
    
    try:
        if args.verbose:
            print(f"Connected to Weaviate: {analyzer.weaviate_url}")
            print(f"Using model: {analyzer.model}")
        
        # Parse topic list
        topics_list = None
        if args.topics:
            topics_list = [t.strip() for t in args.topics.split(",")]
            print(f"Tracking specific topics: {topics_list}")
        
        # Run analysis based on arguments
        if args.emerging:
            # Emerging topics analysis
            recent_years = [int(y.strip()) for y in args.recent_years.split(",")]
            comparison_years = [int(y.strip()) for y in args.comparison_years.split(",")]
            
            print(f"Analyzing emerging topics...")
            print(f"  Recent years: {recent_years}")
            print(f"  Comparison years: {comparison_years}")
            print(f"  Growth threshold: {args.threshold}")
            
            result = analyzer.identify_emerging_topics(
                recent_years=recent_years,
                comparison_years=comparison_years,
                threshold=args.threshold
            )
            
            # Save result
            analyzer.save_report(result, args.output)
            
            # Print summary
            if 'emerging_topics' in result:
                emerging = result['emerging_topics']
                print(f"\nFound {len(emerging)} emerging topics:")
                for i, (topic, data) in enumerate(list(emerging.items())[:10], 1):
                    status = "NEW" if data['is_new'] else "GROWING"
                    print(f"  {i}. {topic} ({data['growth_rate']:.1%} growth) [{status}]")
        
        elif args.relationships:
            # Topic relationships analysis
            print(f"Analyzing topic relationships for year {args.relationships}...")
            
            result = analyzer.analyze_topic_relationships(
                year=args.relationships,
                min_cooccurrence=args.min_cooccurrence
            )
            
            # Save result
            analyzer.save_report(result, args.output)
            
            # Print summary
            if 'centrality' in result:
                centrality = result['centrality']
                print(f"\nTop connected topics:")
                top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                for i, (topic, connections) in enumerate(top_central, 1):
                    print(f"  {i}. {topic} ({connections} connections)")
        
        else:
            # Comprehensive trend analysis
            year_range = (args.start_year, args.end_year)
            print(f"Generating comprehensive trend report for {year_range}...")
            
            # Limit data in test mode
            if args.test_mode:
                print("Running in test mode with limited data...")
                # Modify the analyzer to limit data in test mode
                # This would require adding test mode to the analyzer class
            
            result = analyzer.generate_trend_report(
                year_range=year_range,
                key_topics=topics_list
            )
            
            # Save result
            analyzer.save_report(result, args.output)
            
            # Print summary
            print(f"\n=== ANALYSIS SUMMARY ===")
            print(f"Year Range: {result['year_range']}")
            print(f"Total Papers: {result['total_papers']}")
            print(f"Key Topics: {', '.join(result['key_topics'][:5])}...")
            
            if 'emerging_topics' in result and 'emerging_topics' in result['emerging_topics']:
                emerging = result['emerging_topics']['emerging_topics']
                print(f"\nTop Emerging Topics:")
                for i, (topic, data) in enumerate(list(emerging.items())[:5]):
                    print(f"  {i+1}. {topic} (growth: {data['growth_rate']:.2%})")
        
        print(f"\n✅ Analysis complete! Report saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    finally:
        analyzer.close()

if __name__ == "__main__":
    main() 