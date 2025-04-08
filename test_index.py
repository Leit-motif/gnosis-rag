import requests
import json
import time

def test_index():
    url = "http://127.0.0.1:8000/index"
    headers = {
        "accept": "application/json"
    }
    
    print("Testing /index endpoint...")
    start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers)
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            print("\nSuccess!")
            print(json.dumps(response.json(), indent=2))
        else:
            print("\nError Response:")
            print(json.dumps(response.json(), indent=2))
            
    except Exception as e:
        print(f"\nError: {str(e)}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    test_index() 