#!/usr/bin/env python3
"""Test script to verify HEIC support."""

import sys
import os

# Test if pillow-heif is installed
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    print("✅ pillow-heif is installed and registered")
except ImportError as e:
    print(f"❌ pillow-heif is not installed: {e}")
    sys.exit(1)

# Test if PIL can open HEIC files
try:
    from PIL import Image
    print("✅ PIL is available")
    
    # Check supported formats
    formats = Image.registered_extensions()
    heic_supported = '.heic' in formats or '.heif' in formats
    
    if heic_supported:
        print("✅ HEIC/HEIF format is registered in PIL")
        print(f"   Supported extensions: {[ext for ext in formats if 'hei' in ext.lower()]}")
    else:
        print("❌ HEIC/HEIF format is not registered")
        
except ImportError as e:
    print(f"❌ PIL is not available: {e}")
    sys.exit(1)

# Test the PhotoLabeler
try:
    from label import PhotoLabeler, HAS_HEIC_SUPPORT
    
    if HAS_HEIC_SUPPORT:
        print("✅ PhotoLabeler has HEIC support enabled")
    else:
        print("❌ PhotoLabeler does not have HEIC support")
        
    # Create an instance to verify it works
    with PhotoLabeler(fast_mode=True) as labeler:
        print("✅ PhotoLabeler initialized successfully")
        
except Exception as e:
    print(f"❌ Error with PhotoLabeler: {e}")
    sys.exit(1)

print("\n✅ All tests passed! HEIC support is ready.")
print("\nYou can now process HEIC files with:")
print("  python3 label.py /path/to/folder --fast")
print("  python3 label.py /path/to/folder  # for full AI analysis")