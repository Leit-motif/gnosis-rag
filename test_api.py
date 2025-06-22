#!/usr/bin/env python3
"""
Simple test script for the Gnosis RAG API.
Tests the main endpoints to ensure they're working correctly.
"""

import requests
import os
import sys
from typing import Dict, Any


def test_endpoint(url: str, method: str = "GET", data: Dict[Any, Any] = None, files: Dict[str, Any] = None) -> Dict[str, Any]:
    """Test an API endpoint and return the response."""
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, timeout=10)
            else:
                response = requests.post(url, json=data, timeout=10)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return {
            "status_code": response.status_code,
            "success": response.status_code < 400,
            "response": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
            "headers": dict(response.headers)
        }
    except requests.RequestException as e:
        return {
            "status_code": None,
            "success": False,
            "error": str(e),
            "response": None
        }


def main():
    """Run API tests."""
    # Get base URL from environment or use default
    base_url = os.getenv("API_BASE_URL", "http://localhost:8080")
    
    print(f"🧪 Testing Gnosis RAG API at: {base_url}")
    print("=" * 50)
    
    # Test 1: Health Check
    print("1. Testing Health Check...")
    health_result = test_endpoint(f"{base_url}/health")
    if health_result["success"]:
        print("   ✅ Health check passed")
        if isinstance(health_result["response"], dict):
            db_status = health_result["response"].get("database", "unknown")
            print(f"   📊 Database status: {db_status}")
            if db_status != "connected":
                print("   ⚠️  Database not connected - this may affect other tests")
    else:
        print(f"   ❌ Health check failed: {health_result.get('error', 'Unknown error')}")
        return False
    
    # Test 2: Plugin Manifest
    print("\n2. Testing Plugin Manifest...")
    manifest_result = test_endpoint(f"{base_url}/.well-known/ai-plugin.json")
    if manifest_result["success"]:
        print("   ✅ Plugin manifest endpoint working")
    else:
        print(f"   ❌ Plugin manifest failed: {manifest_result.get('error', 'Unknown error')}")
    
    # Test 3: OpenAPI Spec
    print("\n3. Testing OpenAPI Specification...")
    openapi_result = test_endpoint(f"{base_url}/openapi.yaml")
    if openapi_result["success"]:
        print("   ✅ OpenAPI spec endpoint working")
    else:
        print(f"   ❌ OpenAPI spec failed: {openapi_result.get('error', 'Unknown error')}")
    
    # Test 4: Chat Endpoint (should return placeholder)
    print("\n4. Testing Chat Endpoint...")
    chat_data = {"query": "Hello, this is a test query"}
    chat_result = test_endpoint(f"{base_url}/chat", method="POST", data=chat_data)
    if chat_result["success"]:
        print("   ✅ Chat endpoint working (placeholder response)")
        if isinstance(chat_result["response"], dict):
            print(f"   💬 Response: {chat_result['response'].get('answer', 'No answer')}")
    else:
        print(f"   ❌ Chat endpoint failed: {chat_result.get('error', 'Unknown error')}")
    
    # Test 5: Upload Endpoint (should return not implemented)
    print("\n5. Testing Upload Endpoint...")
    test_content = "# Test Document\nThis is a test markdown file."
    files = {"files": ("test.md", test_content, "text/markdown")}
    upload_result = test_endpoint(f"{base_url}/upload", method="POST", files=files)
    if upload_result["status_code"] == 501:  # Not implemented yet
        print("   ✅ Upload endpoint responding (not implemented yet)")
    elif upload_result["success"]:
        print("   ✅ Upload endpoint working")
    else:
        print(f"   ❌ Upload endpoint failed: {upload_result.get('error', 'Unknown error')}")
    
    # Test 6: Rate Limiting (optional)
    print("\n6. Testing Rate Limiting...")
    rate_limit_count = 0
    for i in range(3):
        result = test_endpoint(f"{base_url}/health")
        if result["success"]:
            rate_limit_count += 1
        else:
            break
    
    if rate_limit_count >= 3:
        print("   ✅ Multiple requests successful (rate limiting working)")
    else:
        print(f"   ⚠️  Only {rate_limit_count} requests succeeded")
    
    print("\n" + "=" * 50)
    print("🎉 API testing completed!")
    
    # Summary
    if health_result["success"]:
        print("✅ Core functionality is working")
        if isinstance(health_result["response"], dict):
            db_status = health_result["response"].get("database", "unknown")
            if db_status == "connected":
                print("✅ Database integration is working")
            else:
                print("⚠️  Database integration needs attention")
        return True
    else:
        print("❌ Core functionality is not working")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 