#!/usr/bin/env python
"""Wrapper script to run model evaluation from project root"""
import os
import sys
os.environ['OTEL_SDK_DISABLED'] = 'true'

# Add project root to path to enable proper imports
sys.path.insert(0, os.path.dirname(__file__))

from src.evaluate import main

if __name__ == "__main__":
    main()