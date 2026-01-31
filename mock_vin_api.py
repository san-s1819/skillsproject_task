#!/usr/bin/env python3
"""
Mock VIN Checker API Server
Simulates a vehicle recall checking service for the assessment
"""

from flask import Flask, request, jsonify
import re

app = Flask(__name__)

# Mock database of VINs and their recall status
KNOWN_VINS = {
    "2HGFC2F59MH123456": {
        "is_affected": True,
        "recall_numbers": ["21V-234"],
        "vehicle_info": {
            "year": 2021,
            "make": "Honda",
            "model": "Accord"
        }
    },
    "5YJ3E1EA5MF123456": {
        "is_affected": True,
        "recall_numbers": ["23V-789"],
        "vehicle_info": {
            "year": 2023,
            "make": "Tesla",
            "model": "Model 3"
        }
    }
}

def validate_vin(vin):
    """Validate VIN format (17 alphanumeric characters, excluding I, O, Q)"""
    if not vin:
        return False
    
    # VIN must be exactly 17 characters
    if len(vin) != 17:
        return False
    
    # VIN should be alphanumeric and not contain I, O, or Q
    vin_pattern = r'^[A-HJ-NPR-Z0-9]{17}$'
    return bool(re.match(vin_pattern, vin.upper()))

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok"}), 200

@app.route('/check-vin', methods=['POST'])
def check_vin():
    """
    Check if a VIN is affected by any recalls
    
    Request body:
    {
        "vin": "17-character VIN"
    }
    
    Response:
    {
        "vin": "...",
        "is_affected": true/false,
        "recall_numbers": [...],
        "vehicle_info": {
            "year": ...,
            "make": "...",
            "model": "..."
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "Invalid request",
                "message": "Request body must be JSON"
            }), 400
        
        vin = data.get('vin', '').strip().upper()
        
        if not vin:
            return jsonify({
                "error": "Missing VIN",
                "message": "VIN parameter is required"
            }), 400
        
        # Validate VIN format
        if not validate_vin(vin):
            return jsonify({
                "error": "Invalid VIN format",
                "message": "VIN must be exactly 17 alphanumeric characters (excluding I, O, Q)"
            }), 400
        
        # Check if VIN is in our known recalls database
        if vin in KNOWN_VINS:
            response_data = {
                "vin": vin,
                **KNOWN_VINS[vin]
            }
            return jsonify(response_data), 200
        else:
            # VIN is valid but not affected by any known recalls
            # Extract basic info from VIN (simplified - real VIN decoding is complex)
            year = extract_year_from_vin(vin)
            make = extract_make_from_vin(vin)
            
            return jsonify({
                "vin": vin,
                "is_affected": False,
                "recall_numbers": [],
                "vehicle_info": {
                    "year": year,
                    "make": make,
                    "model": "Unknown"
                }
            }), 200
            
    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

def extract_year_from_vin(vin):
    """
    Extract model year from VIN (simplified)
    10th character represents the model year
    """
    year_codes = {
        'M': 2021, 'N': 2022, 'P': 2023, 'R': 2024,
        'S': 2025, 'T': 2026, 'V': 2027, 'W': 2028,
        'X': 2029, 'Y': 2030
    }
    
    if len(vin) >= 10:
        year_char = vin[9]
        return year_codes.get(year_char, 2020)
    return 2020

def extract_make_from_vin(vin):
    """
    Extract manufacturer from VIN (very simplified)
    Real WMI (World Manufacturer Identifier) decoding is much more complex
    """
    wmi_prefix = vin[:3] if len(vin) >= 3 else ""
    
    # Simplified mapping of common WMI codes
    wmi_map = {
        '1HG': 'Honda',
        '2HG': 'Honda',
        '19U': 'Honda',
        '4T1': 'Toyota',
        '5YJ': 'Tesla',
        '1FT': 'Ford',
        '1GC': 'Chevrolet',
        '3N1': 'Nissan',
        'KM8': 'Hyundai',
        '5NP': 'Hyundai'
    }
    
    return wmi_map.get(wmi_prefix, "Unknown")

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API documentation"""
    return jsonify({
        "service": "Mock VIN Recall Checker API",
        "version": "1.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/check-vin": "POST - Check VIN for recalls",
            "/": "GET - This documentation"
        },
        "example_request": {
            "method": "POST",
            "url": "/check-vin",
            "body": {
                "vin": "2HGFC2F59MH123456"
            }
        },
        "test_vins": {
            "2HGFC2F59MH123456": "2021 Honda Accord - Has recall",
            "5YJ3E1EA5MF123456": "2023 Tesla Model 3 - Has recall",
            "1HGCM82633A123456": "Valid VIN - No recalls"
        }
    }), 200

if __name__ == '__main__':
    print("=" * 60)
    print("Mock VIN Checker API Server")
    print("=" * 60)
    print("Starting server on http://localhost:8000")
    print("\nTest VINs:")
    print("  2HGFC2F59MH123456 - 2021 Honda Accord (has recall)")
    print("  5YJ3E1EA5MF123456 - 2023 Tesla Model 3 (has recall)")
    print("  Any other valid 17-char VIN - No recalls")
    print("\nTest command:")
    print('  curl -X POST http://localhost:8000/check-vin \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"vin": "2HGFC2F59MH123456"}\'')
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8000, debug=False)
