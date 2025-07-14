# Topic Evolution Analysis Module

This module provides comprehensive tools for analyzing the evolution of research topics in academic conferences over time.

## Quick Start

### From Root Directory
```bash
# Install dependencies
pip install -r analysis/requirements.txt

# Run analysis (from project root)
python analyze_topics.py --start-year 2020 --end-year 2024

# Test the analyzer
python -m analysis.test_topic_analyzer
```

### From Analysis Directory
```bash
cd analysis

# Run CLI directly
python run_topic_analysis.py --start-year 2020 --end-year 2024

# Test functionality
python test_topic_analyzer.py
```

## Components

- **`topic_evolution_analyzer.py`**: Core analysis engine
- **`test_topic_analyzer.py`**: Test suite and visualizations
- **`run_topic_analysis.py`**: Command-line interface
- **`requirements.txt`**: Module dependencies

## Usage Examples

```python
from analysis.topic_evolution_analyzer import TopicEvolutionAnalyzer

# Initialize analyzer
analyzer = TopicEvolutionAnalyzer()

# Generate comprehensive report
report = analyzer.generate_trend_report((2020, 2024))

# Track specific topics
trends = analyzer.track_topic_trends(
    ["transformer", "recommendation", "machine learning"], 
    (2022, 2024)
)

# Identify emerging topics
emerging = analyzer.identify_emerging_topics(
    recent_years=[2023, 2024],
    comparison_years=[2020, 2021]
)
```

See the main project README for detailed documentation. 