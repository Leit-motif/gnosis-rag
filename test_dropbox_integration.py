#!/usr/bin/env python3
"""
Test script for Dropbox integration with backend endpoints.
"""

import requests
from datetime import datetime

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_CONVERSATION_NAME = "Test Dropbox Integration"
TEST_CONTENT = f"This is a test conversation saved at {datetime.now().isoformat()}"


def test_save_endpoint():
    """Test the /save endpoint with Dropbox integration."""
    print("Testing /save endpoint with Dropbox integration...")
    
    # Test data
    save_data = {
        "conversation_name": TEST_CONVERSATION_NAME,
        "content": TEST_CONTENT
    }
    
    try:
        response = requests.post(f"{BASE_URL}/save", json=save_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Save successful: {result['message']}")
            print(f"📁 File path: {result['file_path']}")
            return result['file_path']
        else:
            print(f"❌ Save failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Save request failed: {e}")
        return None


def test_load_endpoint(file_path):
    """Test the /load endpoint with Dropbox integration."""
    print(f"\nTesting /load endpoint for file: {file_path}")
    
    load_data = {
        "file_path": file_path
    }
    
    try:
        response = requests.post(f"{BASE_URL}/load", json=load_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Load successful from {result['source']}")
            print(f"📄 Content preview: {result['content'][:100]}...")
            print(f"🕒 Last modified: {result.get('last_modified', 'Unknown')}")
            return True
        else:
            print(f"❌ Load failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Load request failed: {e}")
        return False


def test_health_endpoint():
    """Test the /health endpoint to ensure the backend is running."""
    print("Testing /health endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Backend is healthy: {result['status']}")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Backend is not accessible: {e}")
        return False


def main():
    """Run the integration tests."""
    print("🧪 Starting Dropbox Integration Tests")
    print("=" * 50)
    
    # Test backend health
    if not test_health_endpoint():
        print("\n❌ Backend is not accessible. Please start the backend server first.")
        print("Run: python backend/main.py")
        return
    
    # Test save functionality
    file_path = test_save_endpoint()
    if not file_path:
        print("\n❌ Save test failed. Cannot proceed with load test.")
        return
    
    # Test load functionality
    if test_load_endpoint(file_path):
        print("\n✅ All tests passed! Dropbox integration is working.")
    else:
        print("\n❌ Load test failed.")
    
    print("\n" + "=" * 50)
    print("🧪 Test Summary Complete")


if __name__ == "__main__":
    main() 