# tabulation.py
# Alice Zhang 
# goal: tabulate data from depression-sampled.csv dataset

# Tabulate: 
#     Total number of posts
#     Total number of unique authors
#     Average post length (measured in word count)
#     Date range of the dataset
#     Top 20 most important words in the posts

import pandas
import re
from datetime import datetime
from gensim.summarization import keywords

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
    
    # avg length of posts
    posts_txt = ""
    sum = 0
    for i in range(len(posts_clean)):
        post = str(posts_clean[i])
        post = post.split(" ")
        if post != "[deleted]" and post != "[removed]":
            sum += len(post)
            posts_txt += post
    
    # date range
    dates = data["created_utc"]
    dt_dates = []
    for i in range(len(dates)):
        if "http" not in dates[i]: # skip over invalid dates/erroring data
            dt_dates.append(datetime.fromtimestamp(int(dates[i])))
    dt_dates.sort()

    # 20 most important posts
    print(keywords(posts_txt))
    
    # print("There are", num_posts, "posts in the dataset.\n")
    # print("There are", num_deleted, "deleted posts in the dataset.\n")
    # print("There are", num_removed, "removed posts in the dataset.\n")
    # print("There are", len(unique_authors), "unique authors in the dataset.\n")
    # print("The average post length is {:0.2f} words.\n".format(sum / (num_posts - num_deleted - num_removed)))
    # print("The date range of the datset is:\n{} to {}.\n".format(dt_dates[0], dt_dates[-1]))

if __name__ == "__main__":
    tabulate('depression-sampled.csv')
