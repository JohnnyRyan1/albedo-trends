#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Format NAO and GBI

https://psl.noaa.gov/data/timeseries/daily/GBI/ 

https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/nao.shtml

"""

# Import packages
import pandas as pd

# Define user
user = 'jr555'

# Define path
path = '/Users/' + user + '/Library/CloudStorage/OneDrive-DukeUniversity/research/albedo/data/'

# Read
gbi = pd.read_csv(path + 'indices/gbi.csv', index_col='Date', parse_dates=['Date'])

# Resample summer
monthly_gbi = gbi.resample('ME').mean()
summer_gbi = monthly_gbi[monthly_gbi.index.month.isin([6, 7, 8])]
summer_gbi = summer_gbi.resample('YE').mean()
summer_gbi.to_csv(path + 'indices/gbi-summer.csv')

#%%


