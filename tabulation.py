# tabulation.py
# Alice Zhang 
# goal: tabulate data from depression-sampled.csv dataset

# Tabulate: 
#     Total number of posts
#     Total number of unique authors
#     Average post length (measured in word count)
#     Date range of the dataset
#     Top 20 most important words in the posts (selftext

import pandas
import re
import datetime
import time

def tabulate(filename):
    data = pandas.read_csv(filename, header=0)
    data_deleted = data[data["selftext"] == "[deleted]"]
    data_removed = data[data["selftext"] == "[removed]"]
    # vals = ["[deleted]", "[removed]"]
    # data_clean = data[data["selftext"].isin(vals) == False] 
    # # reference: https://www.statology.org/pandas-drop-rows-with-value/
    unique_authors = data["author"].unique()
    num_posts = len(data)
    num_deleted = len(data_deleted)
    num_removed = len(data_removed)
    posts_clean = data["selftext"].str.replace(r'[^\w\s]', '', flags=re.UNICODE) 
    # reference: https://stackoverflow.com/questions/47464658/python-efficient-way-to-remove-emojis-and-some-punctuation-from-a-large-dataset
    sum = 0
    for i in range(len(posts_clean)):
        post = str(posts_clean[i])
        post = post.split(" ")
        if post != "[deleted]" and post != "[removed]":
            sum += len(post)
    
    # print("There are", num_posts, "posts in the dataset.\n")
    # print("There are", num_deleted, "deleted posts in the dataset.\n")
    # print("There are", num_removed, "removed posts in the dataset.\n")
    # print("There are", len(unique_authors), "unique authors in the dataset.\n")
    # print("The average post length is {:0.2f} words.\n".format(sum / (num_posts - num_deleted - num_removed)))


if __name__ == "__main__":
    tabulate('depression-sampled.csv')
