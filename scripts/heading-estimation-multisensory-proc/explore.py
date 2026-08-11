# Initialize required libraries
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Download the data file from the github repo
# 1. Get the data path
repo_root = Path.cwd()
data_path = repo_root / "data" / "heading-estimation-multisensory-proc"
# 2. Make the directory if it does not exist already
data_path.mkdir(parents=True, exist_ok=True)
# 3. Download the data file to data_path
data_url = "https://raw.githubusercontent.com/sharootonian/CombinationAndCompetitionHeadingDirection/main/data/data.csv"
response = requests.get(data_url)
(data_path / "data.csv").write_bytes(response.content)

