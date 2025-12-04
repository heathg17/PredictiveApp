#!/usr/bin/env python3
"""Test fluorescence API endpoint"""

import requests
import json

url = "http://localhost:8001/api/predict"
payload = {
    "concentrations": {
        "GXT": 10,
        "BiVaO4": 5,
        "PG": 0,
        "PearlB": 0
    },
    "thickness": 8
}

print("Testing Fluorescence API...")
print(f"URL: {url}")
print(f"Payload: {json.dumps(payload, indent=2)}")
print()

try:
    response = requests.post(url, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response:")
    result = response.json()
    print(json.dumps(result, indent=2))

    if "fluorescence_area" in result:
        print(f"\n✓ Fluorescence prediction working!")
        print(f"  Fluorescence Area: {result['fluorescence_area']:.4f}")
        print(f"  Model Version: {result['model_version']}")
    else:
        print("\n✗ Fluorescence field missing!")

except Exception as e:
    print(f"Error: {e}")
