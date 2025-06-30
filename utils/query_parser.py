import re
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_topic(query: str):
    """
    Extracts the most relevant noun chunk from the query to use as a topic hint.
    """
    doc = nlp(query)
    if doc.noun_chunks:
        # Pick the longest noun chunk (more descriptive)
        return max((chunk.text for chunk in doc.noun_chunks), key=len).strip().lower()
    return query.lower()

def parse_query(query: str):
    """
    Parses the user's natural language query into structured components:
    - year filter (exact/from/to)
    - topic hint (noun chunk)
    """
    # Extract year(s)
    years = re.findall(r'(19|20)\d{2}', query)
    years = list(map(int, years))

    # Determine year filter type
    if "to" in query or "-" in query:
        year_filter = {"from": min(years), "to": max(years)} if len(years) >= 2 else None
    elif "since" in query or "after" in query:
        year_filter = {"from": years[0]} if years else None
    elif "before" in query or "until" in query:
        year_filter = {"to": years[0]} if years else None
    elif years:
        year_filter = {"exact": years[0]}
    else:
        year_filter = None

    # Extract topic
    topic_hint = extract_topic(query)

    return {
        "raw_query": query,
        "year_filter": year_filter,
        "topic_hint": topic_hint
    }
