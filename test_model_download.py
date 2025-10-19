#!/usr/bin/env python3
"""
Test script to demonstrate the automatic model download functionality.
This script tests downloading a small model to verify the downloader works.
"""

import os
import sys
from utils.model_downloader import download_model_if_needed, verify_model_integrity


def test_model_download():
    """Test downloading a small model."""

    # Use a small model for testing
    test_model = "microsoft/DialoGPT-small"
    print(f"Testing model download with: {test_model}")

    try:
        # Test the download function
        local_path = download_model_if_needed(test_model)
        print(f"✓ Model downloaded to: {local_path}")

        # Test integrity verification
        if verify_model_integrity(local_path):
            print("✓ Model integrity verified")
        else:
            print("✗ Model integrity check failed")
            return False

        print("✓ All tests passed!")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_existing_model():
    """Test with an existing local model path."""

    # Test with a path that might exist
    existing_path = "./test_model"

    try:
        result_path = download_model_if_needed(existing_path)
        print(f"✓ Existing model check completed: {result_path}")
        return True
    except Exception as e:
        print(f"✗ Existing model test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing automatic model download functionality...")
    print("=" * 50)

    # Test 1: Download a new model
    print("\n1. Testing model download:")
    success1 = test_model_download()

    # Test 2: Check existing model
    print("\n2. Testing existing model check:")
    success2 = test_existing_model()

    print("\n" + "=" * 50)
    if success1 and success2:
        print("✓ All tests passed! The model downloader is working correctly.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        sys.exit(1)
