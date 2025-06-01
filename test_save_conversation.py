#!/usr/bin/env python3
"""
Simple test script to verify your Obsidian save functionality works.
"""

import requests
import json
import sys
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"

def test_query_functionality():
    """Test that the query endpoint works"""
    print("🔍 Testing query functionality...")
    
    try:
        response = requests.get(
            f"{BASE_URL}/query",
            params={"q": "test query"}
        )
        
        if response.status_code == 200:
            print("✅ Query endpoint works!")
            return True
        else:
            print(f"❌ Query failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False

def test_save_content():
    """Test the save content endpoint"""
    print("\n💾 Testing save content functionality...")
    
    # Example of exactly what you'd send from ChatGPT
    test_content = f"""## ChatGPT Conversation: Problem Solving Session
_Saved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_

**User:** I'm having trouble with my Python script. It keeps throwing a KeyError.

**Assistant:** I can help you debug that KeyError. Can you show me the specific code that's causing the issue?

**User:** Here's the problematic line: `result = data['missing_key']`

**Assistant:** The issue is that 'missing_key' doesn't exist in your data dictionary. You can fix this by using `data.get('missing_key', default_value)` instead.

This test was run at {datetime.now().isoformat()}"""
    
    payload = {
        "conversation_name": f"Test Save {datetime.now().strftime('%H:%M')}",
        "content": test_content
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/save_exact_content",
            json=payload
        )
        
        if response.status_code == 200:
            print("✅ Save content works!")
            result = response.json()
            print(f"   Saved to: {result.get('file_path', 'unknown location')}")
            return True
        else:
            print(f"❌ Save content failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False

def main():
    """Run the tests"""
    print(f"🚀 Testing Obsidian integration against {BASE_URL}")
    print("=" * 50)
    
    query_works = test_query_functionality()
    save_works = test_save_content()
    
    print("\n" + "=" * 50)
    print("📊 Results:")
    print(f"   Query: {'✅ PASS' if query_works else '❌ FAIL'}")
    print(f"   Save:  {'✅ PASS' if save_works else '❌ FAIL'}")
    
    if query_works and save_works:
        print("\n🎉 Everything works! You can now save ChatGPT conversations to Obsidian.")
        print("\nTo use: Send your conversation content to the 'save_conversation_save_conversation_post' tool")
    else:
        print("\n⚠️  Some functionality failed:")
        if not save_works:
            print("   - Check if your server is running")
            print("   - Verify vault path in config.json")
            print("   - Ensure write permissions to vault directory")
    
    return 0 if (query_works and save_works) else 1

if __name__ == "__main__":
    sys.exit(main()) 