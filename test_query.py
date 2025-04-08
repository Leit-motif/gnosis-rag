import requests
import json
import time
from datetime import datetime, timedelta

def test_query():
    # Query parameters
    params = {
        "q": "How do my thoughts on transhumanism connect to my views on consciousness and spirituality?",
        "use_graph": True,  # Enable graph-based retrieval
        "k": 5  # Number of results to return
    }
    
    print("Query:", params["q"])
    print("Using graph-based retrieval:", params["use_graph"])
    
    start_time = time.time()
    
    try:
        response = requests.get("http://127.0.0.1:8000/query", params=params)
        print("\nStatus Code:", response.status_code)
        
        if response.status_code == 200:
            result = response.json()
            print("\nResponse:", result["response"])
            print("\nSources:")
            
            for i, source in enumerate(result["sources"], 1):
                print(f"\nSource {i}:")
                print(f"Score: {source['score']:.4f}")
                print(f"File: {source['metadata'].get('source', 'N/A')}")
                print(f"Date: {source['metadata'].get('date', 'N/A')}")
                print(f"Tags: {', '.join(source['metadata'].get('tags', []))}")
                print(f"Graph Distance: {source['metadata'].get('graph_distance', 'N/A')}")
                print(f"Graph Similarity: {source['metadata'].get('graph_similarity', 'N/A')}")
                print(f"Excerpt: {source['excerpt'][:500]}...")
        else:
            print("Error!")
            print(response.text)
            
    except Exception as e:
        print(f"Error: {str(e)}")
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f} seconds")

if __name__ == "__main__":
    test_query() 