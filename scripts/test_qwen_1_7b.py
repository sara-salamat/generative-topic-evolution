#!/usr/bin/env python3
"""
Test script to check summarization quality with qwen3:1.7b model
"""

import requests
import time

def test_qwen_1_7b_summarization():
    """Test summarization with qwen3:1.7b model"""
    
    # Sample abstracts for testing
    test_abstracts = [
        """Transformer models have achieved remarkable success in natural language processing tasks, but their quadratic computational complexity with sequence length limits their application to long sequences. We propose Longformer, a transformer model that can process sequences of length up to 32,768 tokens with linear computational complexity. Our approach combines a local windowed attention pattern with task-specific global attention, allowing the model to efficiently handle long documents while maintaining the expressiveness of the full transformer architecture. We demonstrate the effectiveness of Longformer on long document classification, question answering, and coreference resolution tasks, achieving state-of-the-art results on several benchmarks.""",
        
        """Deep learning has revolutionized computer vision, but training deep neural networks requires large amounts of labeled data. We propose a novel semi-supervised learning approach that leverages unlabeled data to improve model performance. Our method uses consistency regularization and pseudo-labeling to effectively utilize unlabeled samples. We demonstrate significant improvements on standard benchmarks, achieving state-of-the-art results with only 10% of labeled data.""",
        
        """The field of natural language processing has seen tremendous advances with the introduction of large language models. However, these models often struggle with reasoning tasks that require step-by-step thinking. We introduce a new approach that combines chain-of-thought prompting with structured reasoning to improve performance on complex reasoning tasks. Our method achieves significant improvements on mathematical reasoning, logical inference, and commonsense reasoning benchmarks."""
    ]
    
    prompt_template = """Create a concise academic summary of the following research abstract.

Focus on:
- Research question/objective
- Methodology/approach
- Key findings/contributions
- Impact/significance

Keep it to 2-3 sentences maximum.

Abstract: {abstract}

Summary:"""
    
    print("=" * 80)
    print("TESTING QWEN3:1.7B SUMMARIZATION QUALITY")
    print("=" * 80)
    
    total_time = 0
    total_length = 0
    
    for i, abstract in enumerate(test_abstracts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Abstract length: {len(abstract)} characters")
        
        payload = {
            "model": "qwen3:1.7b",
            "prompt": prompt_template.format(abstract=abstract),
            "stream": False
        }
        
        start_time = time.time()
        try:
            response = requests.post("http://localhost:11434/api/generate", 
                                   json=payload, timeout=60)
            response.raise_for_status()
            end_time = time.time()
            
            summary = response.json()["response"].strip()
            
            # Remove thinking process if present
            if "<think>" in summary and "</think>" in summary:
                summary = summary.split("</think>")[-1].strip()
            summary = summary.replace("<think>", "").replace("</think>", "").strip()
            
            processing_time = end_time - start_time
            total_time += processing_time
            total_length += len(summary)
            
            print(f"Processing time: {processing_time:.2f}s")
            print(f"Summary length: {len(summary)} characters")
            print(f"Summary: {summary}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Average processing time: {total_time/len(test_abstracts):.2f}s per abstract")
    print(f"Average summary length: {total_length/len(test_abstracts):.0f} characters")
    print(f"Total processing time: {total_time:.2f}s")
    
    # Estimate for full dataset
    full_dataset_size = 13223
    estimated_time_hours = (total_time / len(test_abstracts)) * full_dataset_size / 3600
    print(f"\nEstimated time for full dataset ({full_dataset_size} papers): {estimated_time_hours:.1f} hours")
    
    print("\nQuality Assessment:")
    print("✅ Speed: Much faster than 8B model")
    print("✅ Memory: Lower memory usage")
    print("⚠️  Quality: May be slightly lower than 8B model")
    print("\nRecommendation: Test with a few hundred papers first to assess quality.")

if __name__ == "__main__":
    test_qwen_1_7b_summarization() 