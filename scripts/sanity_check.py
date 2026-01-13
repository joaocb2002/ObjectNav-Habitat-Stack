"""
Sanity Check Script for Habitat Environment in Docker Containers

This script is designed to be run within Docker containers to verify that the 
Habitat-Sim and Habitat-Lab packages are properly installed and can be imported 
successfully. It serves as a quick diagnostic tool to confirm the environment is 
correctly configured and ready to run from whichever machine the container is 
deployed on, ensuring portability across different systems.

Usage:
    python scripts/sanity_check.py

The script will print the Python version and import status for both habitat_sim 
and habitat packages. If any imports fail, an exception will be raised with 
detailed error information, indicating the container environment needs attention 
before running training or evaluation tasks.
"""

import sys

def main():
    print("Python:", sys.version)
    try:
        import habitat_sim
        print("habitat_sim import: OK")
    except Exception as e:
        print("habitat_sim import: FAIL")
        raise

    try:
        import habitat
        print("habitat_lab import: OK")
    except Exception:
        print("habitat_lab import: FAIL")
        raise

if __name__ == "__main__":
    main()
