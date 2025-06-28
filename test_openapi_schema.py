#!/usr/bin/env python3
"""
Test script to validate OpenAPI schema
"""
import requests
import json
import yaml


def test_openapi_schema():
    """Test that the OpenAPI schema is valid and accessible"""
    
    # Test the YAML schema directly
    try:
        print("Testing OpenAPI YAML schema...")
        url = "https://gnosis-rag-api.onrender.com/openapi.yaml"
        response = requests.get(url, timeout=30)
        
        print(f"YAML Schema Status: {response.status_code}")
        
        if response.status_code == 200:
            # Try to parse the YAML
            try:
                schema = yaml.safe_load(response.text)
                print("✅ YAML schema is valid")
                print(f"Title: {schema.get('info', {}).get('title', 'N/A')}")
                print(f"Version: {schema.get('info', {}).get('version', 'N/A')}")
                
                # Check if save operation exists
                save_op = schema.get('paths', {}).get('/save', {}).get('post', {})
                if save_op:
                    print("✅ Save operation found in schema")
                    print(f"Operation ID: {save_op.get('operationId', 'N/A')}")
                else:
                    print("❌ Save operation NOT found in schema")
                    
            except yaml.YAMLError as e:
                print(f"❌ YAML parsing failed: {e}")
        else:
            print(f"❌ Failed to fetch YAML schema: {response.text}")
            
    except Exception as e:
        print(f"Error testing YAML schema: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test the JSON schema
    try:
        print("Testing OpenAPI JSON schema...")
        url = "https://gnosis-rag-api.onrender.com/openapi.json"
        response = requests.get(url, timeout=30)
        
        print(f"JSON Schema Status: {response.status_code}")
        
        if response.status_code == 200:
            # Try to parse the JSON
            try:
                schema = response.json()
                print("✅ JSON schema is valid")
                print(f"Title: {schema.get('info', {}).get('title', 'N/A')}")
                print(f"Version: {schema.get('info', {}).get('version', 'N/A')}")
                
                # Check if save operation exists
                save_op = schema.get('paths', {}).get('/save', {}).get('post', {})
                if save_op:
                    print("✅ Save operation found in schema")
                    print(f"Operation ID: {save_op.get('operationId', 'N/A')}")
                    
                    # Check schema components
                    save_request = schema.get('components', {}).get('schemas', {}).get('SaveRequest', {})
                    if save_request:
                        print("✅ SaveRequest schema found")
                        print(f"Required fields: {save_request.get('required', [])}")
                    else:
                        print("❌ SaveRequest schema NOT found")
                else:
                    print("❌ Save operation NOT found in schema")
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
        else:
            print(f"❌ Failed to fetch JSON schema: {response.text}")
            
    except Exception as e:
        print(f"Error testing JSON schema: {e}")


if __name__ == "__main__":
    test_openapi_schema() 