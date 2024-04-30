import pandas as pd
import os, argparse
from pathlib import Path

if __name__ == '__main__':

    # iterate through results recognizer folder and all its sub folders
    results_path = Path("results", "recognizer")
    for root, dirs, files in os.walk(results_path):
        for file in files:
            print(file)

