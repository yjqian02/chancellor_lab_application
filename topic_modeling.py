
import re                           # use re for regex (data cleaning)
import sys                          # use for reading in command line arguments
from datetime import datetime       # use datetime to find time frame
from collections import defaultdict # use collections to have dictionaries with default values
import matplotlib.pyplot as plt     # use matplotlib for visualization
import pandas as pd                 # use pandas to load data
from time import gmtime, strftime   # use time to debug/benchmark code
import pickle                       # use pickle to load dictionary (to save on runtime)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


data = pd.read_csv("depression-sampled.csv", header=0)

cv = CountVectorizer(max_df = 0.90, min_df = 2, stop_words = "english")

cv_fit = cv.fit_transform(data["selftext"])
print("shape", cv_fit)