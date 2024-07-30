import pandas as pd
from pathlib import Path


if __name__ == '__main__':
    # load mappings
    mappings = pd.read_csv(Path("results", "mappings", "realistic_mappings.csv"))
    