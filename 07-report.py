#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Report stats.

"""


# Import packages
import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.stats import ttest_rel
import glob
import os
import matplotlib.pyplot as plt
import pymannkendall as mk

# Define user
user = 'johnnyryan'

# Define path
path = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/'

# Define files
files = sorted(glob.glob(path + 'data/station/*.csv'))

# Import AWS metadata
aws = pd.read_csv(path + 'data/promice/AWS_sites_metadata.csv')

# Import RMSEs
station_df = pd.read_csv(path + 'data/station-error.csv')


#%%

print















