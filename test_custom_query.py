import requests
import json

def run_test(query, date_range=None, tags=None):
    url = "http://localhost:8000/query"
    params = {
        "q": query
    }
    
    # Add optional parameters
    if date_range:
        params["date_range"] = date_range
    if tags:
        params["tags"] = tags
    
    param_str = f"Query: {params['q']}"
    if date_range:
        param_str += f", Date range: {date_range}"
    if tags:
        param_str += f", Tags: {tags}"
    
    print(f"\n--- Testing {param_str} ---")
    response = requests.get(url, params=params)
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\nResponse:")
        print(data["response"])
        
        print("\nNumber of sources:", len(data["sources"]))
        
        if len(data["sources"]) > 0:
            print("\nSources:")
            for i, source in enumerate(data["sources"], 1):
                print(f"\nSource {i}:")
                if "metadata" in source:
                    meta = source["metadata"]
                    if "source" in meta:
                        print(f"File: {meta['source']}")
                    if "path" in meta:
                        print(f"Path: {meta['path']}")
                    if "date" in meta:
                        print(f"Date: {meta['date']}")
                    if "created" in meta:
                        print(f"Created: {meta['created']}")
                    if "tags" in meta and meta["tags"]:
                        print(f"Tags: {', '.join(meta['tags'])}")
                # Print full content for debugging
                if "content" in source:
                    print(f"Content preview: {source['content'][:200]}...")
                    print(f"Content length: {len(source['content'])}")
                if "score" in source:
                    print(f"Score: {source['score']}")
    else:
        print("Error:", response.text)
    
    print("=" * 50)

if __name__ == "__main__":
    # Test very specific query that's similar to what worked in the test_query.py
    print("Running most specific query from example...")
    run_test("How do my thoughts on transhumanism connect to my views on consciousness and spirituality?") 