#!/usr/bin/env python3
"""
Simple entry point for Topic Evolution Analysis
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

from analysis.run_topic_analysis import main

if __name__ == "__main__":
    main() 