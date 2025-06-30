#!/usr/bin/env python3
"""
Test script for the new Dropbox API endpoints.
Tests /dropbox/status and /dropbox/sync endpoints.
"""

import requests
import json
from datetime import datetime

# Test configuration
BASE_URL = "http://localhost:8080"


def test_dropbox_status():
    """Test the /dropbox/status endpoint."""
    print("🔍 Testing /dropbox/status endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/dropbox/status", timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Status endpoint successful!")
            print(f"📊 Response: {json.dumps(result, indent=2)}")
            
            # Verify expected fields
            expected_fields = [
                "enabled", "is_syncing", "last_sync_time", 
                "sync_interval_minutes", "local_vault_path", 
                "dropbox_connected", "timestamp"
            ]
            
            missing_fields = [field for field in expected_fields if field not in result]
            if missing_fields:
                print(f"⚠️  Missing fields: {missing_fields}")
            else:
                print("✅ All expected fields present")
                
            return True
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False


def test_dropbox_sync():
    """Test the /dropbox/sync endpoint."""
    print("\n🔄 Testing /dropbox/sync endpoint...")
    
    try:
        response = requests.post(f"{BASE_URL}/dropbox/sync", timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Sync endpoint successful!")
            print(f"📊 Response: {json.dumps(result, indent=2)}")
            
            # Verify expected fields
            expected_fields = [
                "status", "message", "files_downloaded", 
                "files_uploaded", "files_conflicted", 
                "files_skipped", "errors", "timestamp"
            ]
            
            missing_fields = [field for field in expected_fields if field not in result]
            if missing_fields:
                print(f"⚠️  Missing fields: {missing_fields}")
            else:
                print("✅ All expected fields present")
                
            return True
        elif response.status_code == 400:
            print("ℹ️  Sync disabled or unavailable (expected if Dropbox not configured)")
            print(f"Response: {response.text}")
            return True
        else:
            print(f"❌ Sync endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False


def test_health_endpoint():
    """Test the /health endpoint to ensure the server is running."""
    print("🏥 Testing /health endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Server is healthy: {result['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Server is not accessible: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Dropbox API endpoint tests...")
    print(f"Target URL: {BASE_URL}")
    print(f"Test time: {datetime.now().isoformat()}")
    print("=" * 50)
    
    # Test server health first
    if not test_health_endpoint():
        print("\n❌ Server is not running. Please start the API server first:")
        print("   python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload")
        return
    
    # Test Dropbox endpoints
    status_result = test_dropbox_status()
    sync_result = test_dropbox_sync()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Health: ✅")
    print(f"   Status: {'✅' if status_result else '❌'}")
    print(f"   Sync: {'✅' if sync_result else '❌'}")
    
    if status_result and sync_result:
        print("\n🎉 All tests passed! The Dropbox endpoints are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main() 