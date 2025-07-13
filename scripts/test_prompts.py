import requests
import time
from prompt_examples import *

# Sample abstract for testing
SAMPLE_ABSTRACT = """Transformer models have achieved remarkable success in natural language processing tasks, but their quadratic computational complexity with sequence length limits their application to long sequences. We propose Longformer, a transformer model that can process sequences of length up to 32,768 tokens with linear computational complexity. Our approach combines a local windowed attention pattern with task-specific global attention, allowing the model to efficiently handle long documents while maintaining the expressiveness of the full transformer architecture. We demonstrate the effectiveness of Longformer on long document classification, question answering, and coreference resolution tasks, achieving state-of-the-art results on several benchmarks."""

def test_prompt(prompt_template, abstract, model="qwen3:8b"):
    """Test a prompt template and return the summary"""
    prompt = prompt_template.format(abstract=abstract)
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        return f"Error: {e}"

def compare_prompts():
    """Compare different prompts on the same abstract"""
    prompts = {
        "Basic": BASIC_PROMPT,
        "Improved": IMPROVED_PROMPT,
        "Academic": ACADEMIC_PROMPT,
        "Technical": TECHNICAL_PROMPT,
        "Simple": SIMPLE_PROMPT,
        "Structured": STRUCTURED_PROMPT,
        "Length Constrained": LENGTH_CONSTRAINED_PROMPT
    }
    
    print("=== Testing Different Prompts ===\n")
    print(f"Sample Abstract:\n{SAMPLE_ABSTRACT}\n")
    print("=" * 80)
    
    for name, prompt_template in prompts.items():
        print(f"\n--- {name} Prompt ---")
        print(f"Prompt: {prompt_template.format(abstract='[ABSTRACT]')[:100]}...")
        print(f"\nSummary:")
        
        summary = test_prompt(prompt_template, SAMPLE_ABSTRACT)
        print(summary)
        
        # Add delay between requests
        time.sleep(2)
        print("-" * 40)

if __name__ == "__main__":
    compare_prompts() 