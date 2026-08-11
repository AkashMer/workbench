# Initialize required libraries
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Download the data files from the github repo
# 1. Get the data path
repo_root = Path.cwd()
data_path = repo_root / "data" / "heading-estimation-multisensory-proc"
# 2. Make the directory if it does not exist already
data_path.mkdir(parents=True, exist_ok=True)
# 3. Download the data file to data_path if not already present
if not (data_path / "data.csv").exists():
    data_url = "https://raw.githubusercontent.com/sharootonian/CombinationAndCompetitionHeadingDirection/main/data/data.csv"
    response = requests.get(data_url)
    (data_path / "data.csv").write_bytes(response.content)
if not (data_path / "dataconf.csv").exists():
    dataconf_url = "https://raw.githubusercontent.com/sharootonian/CombinationAndCompetitionHeadingDirection/main/data/dataconf.csv"
    response = requests.get(dataconf_url)
    (data_path / "dataconf.csv").write_bytes(response.content)    

# Load the file and check the structure
bva_data = pd.read_csv(data_path / "data.csv")
print(bva_data.columns)
print(bva_data.shape)

# Number of unique participants
len(bva_data.subj.unique())
# 30 participants, confirmed all participant data present

# Get a general description of the dataframe
bva_data.describe(include='all')
# Max 399 trials per participant
# target: seems uniformly distribution from 0 - 360
# respond makes sense to be > 360 but then error is between -180 to 180: needs further investigation
# trial_duration: most likely represents the duration of whole trial, no way to get response time (needs confirmation)
# I wanna confirm if fb and fb_true match and when offset is zero
# fb_time: Most likely time from start to onset of feedback image

# Load and get the description of dataconf file as well
bva_dataconf = pd.read_csv(data_path / "dataconf.csv")
bva_dataconf.describe(include='all')
# Only two new columns
# conf: A post-hoc cone angle asked from the participant for the door position in their response view
# viewAmount: needs confirmation but the view angle traversed while feedback was visible
# ie. serves as a measure of how much attention was payed to the feedback

# How does high respond correspond to error values?
bva_data.query("respond > 500")
# error = respond - target
(bva_data.respond - bva_data.target - bva_data.error).describe()
# Confirmed => respond is the cumulative angle for the whole trial duration
# Hence, makes sense why > 360

# Confirm trial duration is the whole trial
sns.histplot(bva_data, x='trial_duration') # right-skewed in msec
# Is fb_time < trial_duration always?
(bva_data.fb_time < bva_data.trial_duration).mean() # not 1
# Check which fb_time > trial_duration
bva_data.query("fb_time > trial_duration") # zero rows
# Maybe equal?
bva_data.query("fb_time == trial_duration")
# trial duration is start of encoding period to response button press

# Do fb and fb_true match when offset is zero
(bva_data.eval("fb_diff = fb - fb_true")
    .query("condition == 'FB' & fb_offset == 0")
    .describe()
)
# fb_offset is never zero
# On further confirmation, fb_offset is drawn two ways:
# 1. Gaussian with 0 mean and SD: 30 degrees (70% trials/participant)
# 2. Uniformly from -180 to 180 (30% trials/participant)

# Confirm viewAmount is zero for noFB trials
(
    bva_dataconf.query("condition == 'noFB'")
        .viewAmount.describe()
)
# All zeros, confirmed
