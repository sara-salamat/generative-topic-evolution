# Different prompt strategies for abstract summarization

# Current basic prompt
BASIC_PROMPT = """Summarize the following scientific abstract in 2-4 sentences, focusing on the main contributions and findings.

Abstract: {abstract}

Summary:"""

# Improved prompt with specific instructions
IMPROVED_PROMPT = """You are a research assistant tasked with creating concise summaries of scientific papers. 

Please summarize the following abstract in 2-3 sentences, focusing on:
1. The main research problem or objective
2. The key methodology or approach used
3. The primary findings or contributions

Abstract: {abstract}

Summary:"""

# Academic-focused prompt
ACADEMIC_PROMPT = """Create a concise academic summary of the following research abstract.

Focus on:
- Research question/objective
- Methodology/approach
- Key findings/contributions
- Impact/significance

Keep it to 2-3 sentences maximum.

Abstract: {abstract}

Summary:"""

# Technical prompt for ML/AI papers
TECHNICAL_PROMPT = """Summarize this technical paper abstract concisely.

Include:
- Problem being solved
- Technical approach/method
- Results/performance
- Novel contributions

Abstract: {abstract}

Summary:"""

# Simple and direct prompt
SIMPLE_PROMPT = """Summarize this abstract in 2-3 sentences:

{abstract}

Summary:"""

# Structured prompt
STRUCTURED_PROMPT = """Summarize the following abstract:

Problem: [What problem does this research address?]
Method: [What approach/methodology is used?]
Results: [What are the key findings?]

Abstract: {abstract}

Summary:"""

# Prompt with length constraint
LENGTH_CONSTRAINED_PROMPT = """Create a concise summary of this abstract in exactly 2 sentences.

First sentence: State the problem and approach
Second sentence: State the results and significance

Abstract: {abstract}

Summary:"""

# Prompt for different domains
DOMAIN_SPECIFIC_PROMPT = """Summarize this {domain} paper abstract in 2-3 sentences.

Focus on the specific contributions to {domain} research.

Abstract: {abstract}

Summary:"""

# Example usage:
# DOMAIN_SPECIFIC_PROMPT.format(domain="machine learning", abstract="...") 