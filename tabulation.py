# tabulation.py
# Alice Zhang 
# goal: tabulate data from depression-sampled.csv dataset

# Tabulate: 
#     Total number of posts
#     Total number of unique authors
#     Average post length (measured in word count)
#     Date range of the dataset
#     Top 20 most important words in the posts

import re                           # use re for regex (data cleaning)
from datetime import datetime       # use datetime to find time frame
from collections import defaultdict # use collections to have dictionaries with default values
import matplotlib.pyplot as plt     # use matplotlib for visualization
import pandas as pd                 # use pandas to load data
from time import gmtime, strftime   # use time to debug/benchmark code
import pickle                       # use pickle to load dictionary (to save on runtime)

# import nltk
# nltk.download("stopwords")
# from nltk.corpus import stopwords


# global lists for part 5
# altered list from stopwords.words() from nltk 
# "" is to account for first index
common = ["i", "im", "me", "my", "myself", "we", "our", "ours",
    "ourselves", "you", "youre", "yove", "youll", "youd", "else",
    "your", "yours", "yourself", "yourselves", "he", "him",
    "his", "himself", "she", "shes", "her", "hers", "herself","want",
    "it", "its", "its", "itself", "they", "them", "their", "depressed",
    "theirs", "themselves", "what", "which", "who", "whom", "depression",
    "this", "that", "thall", "these", "those", "am", "is",
    "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "a", "an", "what", "or", "just",
    "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "cant",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "dont", "should", "shouldve", "now", "d", "ll", "m", "o",
    "re", "ve", "y", "ain", "aren", "arent", "couldn", "couldnt", "didn", "didnt", "doesn", "doesnt", "hadn",
    "hadnt", "hasn", "hasnt", "haven", "havent", "isn", "isnt", "ma", "mightn", "mightnt", "mustn",
    "mustnt", "needn", "neednt", "shan", "shant", "shouldn", "shouldnt", "wasn", "wasnt", "weren",
    "werent", "won", "wont", "wouldn", "wouldnt", "dont", "ive", "like", "know", "even", "one", "would", "much", 
    "going", "really", "things", "back", "make", "get", "", "could", "still", "something", "go", "got", "people", "way", 
    "feeling", "good", "always", "ever", "since", "say", "getting", "see", "year", "anything"]
    # starter set of 'risk' words from: https://myvocabulary.com/word-list/depression-vocabulary/
d_vocab = '''Ability, Abnormal, Abuse, Adolescents, Affect, Agency, Aid, 
    Alarm, Alienation, All ages, Alone, Anger, Anguish, Antidepressant, Anxiety, 
    Anxious, Attempt, Attention, Attitude, Awareness,
    Bipolar, Blase, Blue, Brain,
    Caregiver, Certify, Child, Clarity, Clinical, Communication, Concern, Conclude, Condition, Confide, Confusion, Cooperative, Cope, Counsel, Courage, Cruel, Cure, Cycle,
    Data, Death, Debilitating, Defeated, Degree, Depressed, Depression, Descent, Despair, Detriment, Diagnosis, Die, Discrimination, Disease, Disinterest, Disorder, Disorder, Distracted, Doctor, Dog days, Down, Drugs,
    Education, Effect, Endure, Esteem, Evaluation,
    Family, Fatigued, Fear, Feelings, Fight, Finality, Friends,
    Gain, Grief, Grieving, Guideline,
    Hard work, Heal, Health, Help, Helpless, Hereditary, Hopelessness, Hot-line, Hurt,
    Immune, Improvement, Inability, Inactivity, Indicator, Insecure, Interested, Interfere, Involvement, Irritable, Isolation, Issues,
    Jeer, Joking,
    Kill, Knowledge, Knowledgeable,
    Label, Lack, Level, Level, Listening,
    Media, Medication, Medicine, Melancholia, Mental, Mental health, Misunderstanding, Monitor,
    Necessary, Need, Negative, Normal, Nothing,
    Observation, Oncoming, Opinion, Option, Organize, Overcome, Overwhelmed,
    Pain, Panic, Parents, Patience, Patient, Pattern, Pay attention, Peers, Personal, Physician, Pills, Prescription, Prevent, Prevention, Programs, Progress, Progressive, Protect, Psychiatrist,
    Quality, Quantity, Query, Quest,
    Reality, Report, Requirement, Resulting, Review,
    Sadness, Scared, Security, Separation, Seriousness, Siblings, Signs, Skills, Sleep, Sleep pattern, Solitary, Sorrow, Source, Statistics, Stigma, Strength, Struggle, Studies, Substance abuse, Succor, Suffer, Suicide, Sympathetic, Symptoms,
    Tack, Talk, Talking, Teenagers, Tentative, Terrified, Therapy, Thoughts, Time, Tired, Tragedy, Tragic, Trajectory, Treat, Treatment, Treatment, Treatment, Triumph, Troubled,
    Uncertain, Uncomfortable, Understanding, Unfulfilled, Unique, Unsettling, Unusual,
    Validation, Victim,
    Warning, Watch, Withdrawal, World Health Organization, Worry, Worthless	YouthZero'''

def tabulate(filename):
    data = pd.read_csv(filename, header=0)
    data_deleted = data[data["selftext"] == "[deleted]"]
    data_removed = data[data["selftext"] == "[removed]"]
    # vals = ["[deleted]", "[removed]"]
    # data_clean = data[data["selftext"].isin(vals) == False] 
    # # reference: https://www.statology.org/pandas-drop-rows-with-value/
    unique_authors = data["author"].unique()
    num_posts = len(data)
    num_deleted = len(data_deleted)
    num_removed = len(data_removed)
    posts_clean = data["selftext"].str.replace(r'[^\w\s]', '', flags=re.UNICODE, regex=True) 
    # reference: https://stackoverflow.com/questions/47464658/python-efficient-way-to-remove-emojis-and-some-punctuation-from-a-large-dataset
    
    print("Load data", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    # avg length of posts
    # sum up lengths of valid posts and 
    # assign each word a weighting based on frequency and
    # whether identification with "risk" words (for top 20 words)
    word_weight = defaultdict(int) 
    sum = 0
    # cast pandas series to array for optimization (this is where the code runs slowest)
    # reference: https://stackoverflow.com/questions/68671852/best-way-to-iterate-through-elements-of-pandas-series
    for i, post in enumerate(posts_clean.to_numpy()):
        post = str(post)
        post = post.split(" ")
        if post != "[deleted]" and post != "[removed]":
            sum += len(post)
            for word in post:
                word_weight[word] += 1
                # increase weight of posts with more comments
                num_com = data["num_comments"][i]
                if not pd.isna(num_com):
                    word_weight[word] += int(num_com)
                # increase weight of posts with polarizing scores
                # (ie very positive or very negative)
                score = data["score"][i]
                if not pd.isna(score):
                    word_weight[word] += abs(int(score))
                # increase weight of risk words
                if word in d_vocab:
                        word_weight[word] += 1 

    print("Avg length", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    # date range
    dates = data["created_utc"]
    dt_dates = []
    for i in range(len(dates)):
        # skip over invalid dates/erroring data
        if "http" not in dates[i]: 
            dt_dates.append(datetime.fromtimestamp(int(dates[i])))    
    dt_dates.sort()

    print("Date range", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    # 20 most important posts
    top_twenty = {}
    count = 1
    # sort word weights from highest to lowest
    word_weight = dict(reversed(sorted(word_weight.items(), key=lambda x:x[1])))
    for word in word_weight.keys():
        if count < 21 and word.lower() not in common:
            top_twenty[count] = word
            count += 1

    print("20 most important", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    ## additional tabulation
    # visulatization of scores vs comments
    # with outliers
    plt.scatter(list(data["num_comments"]),list(data["score"]))
    plt.ylabel("Score")
    plt.xlabel("Number of Comments")
    plt.title("Scores vs Comments Visualized With outliers")
    plt.show()

    # without outliers
    plt.scatter(list(data["num_comments"]),list(data["score"]))
    plt.ylabel("Score")
    plt.xlabel("Number of Comments")
    plt.xlim(0, 400)
    plt.title("Scores vs Comments Visualized Without Outliers")
    plt.show()

    print("Additional Vis\nScores vs Comments", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    # visualization of date vs comments (amplified by score)
    dates = data["created_utc"]
    dt_dates = []
    comments_for_dates = []
    scores_for_amp = []
    for i in range(len(dates)):
        # skip over invalid dates/erroring data
        if "http" not in dates[i] and not pd.isna(data["score"][i]): 
            dt_dates.append(datetime.fromtimestamp(int(dates[i])))
            comments_for_dates.append(int(data["num_comments"][i]))
            scores_for_amp.append(abs(float(data["score"][i])))

    for i in range(len(scores_for_amp)):
        scores_for_amp[i] /= 10

    plt.scatter(comments_for_dates, dt_dates, s = scores_for_amp)
    plt.xlabel("Number of Comments")
    plt.ylabel("Post Creation Date")
    plt.xlim(0, 400)
    plt.title("Post Timeline with Score Amplification")
    plt.show()

    print("Dates vs Comments", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    # visualization of avg # comments and avg scores
    titles = data["title"].str.replace(r'[^\w\s]', '', flags=re.UNICODE)
    title_fq = defaultdict(int)
    for i in range(len(titles)):
        title = str(titles[i])
        title = title.split(" ")
        for word in title:
            if word.lower() not in common:
                title_fq[word] += 1
    title_fq = dict(reversed(sorted(title_fq.items(), key=lambda x:x[1])))
    cats = defaultdict(int)
    cats_com = defaultdict(int)
    cats_score = defaultdict(int)
    vis_data = defaultdict(list)
    i = 0
    for title in title_fq.keys():
        if i < 21:
            cats[title] = 0
            i += 1
    for i in range(len(titles)):
        title = str(titles[i])
        for cat in cats.keys():
            if cat in title:
                cats[cat] += 1
                cats_com[cat] += data["num_comments"][i]
                cats_score[cat] += data["score"][i]
    # calculate average scores and number of comments
    for cat, score in cats_score.items():
        vis_data["Score"].append(score // cats[cat])
    for cat, com in cats_com.items():
        vis_data["Comments"].append(com // cats[cat])

    vis_data = pd.DataFrame(vis_data, index=list(cats.keys()))
    vis_data.plot(kind="bar")
    plt.ylabel("Average Scores/Number of Comments")
    plt.xlabel("Post Category")
    plt.title("Average Scores and  Number of Comments for Posts in Top Categories")
    plt.show()

    print("Comments per Topic", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    print("There are", num_posts, "posts in the dataset.\n")
    print("There are", num_deleted, "deleted posts in the dataset.\n")
    print("There are", num_removed, "removed posts in the dataset.\n")
    print("There are", len(unique_authors), "unique authors in the dataset.\n")
    print("The average post length is {:0.2f} words.\n".format(sum / (num_posts - num_deleted - num_removed)))
    print("The date range of the datset is:\n{} to {}\n".format(dt_dates[0], dt_dates[-1]))
    print("The twenty most important words in the posts are:")
    for i, word in top_twenty.items():
        print(i, ":", word)

if __name__ == "__main__":
    d_vocab = d_vocab.split(',')
    for i in range(len(d_vocab)):
        d_vocab[i] = re.sub(r'\s+', "", d_vocab[i])
        d_vocab[i] = d_vocab[i].lower()

    tabulate('depression-sampled.csv')



## sample output
# There are 30000 posts in the dataset.

# There are 342 deleted posts in the dataset.

# There are 3082 removed posts in the dataset.

# There are 24725 unique authors in the dataset.

# The average post length is 172.75 words.

# The date range of the datset is:
# 2020-10-13 12:22:17 to 2019-08-23 13:47:19

# The twenty most important words in the posts are:
# 1 : feel
# 2 : life
# 3 : time
# 4 : friends
# 5 : never
# 6 : think
# 7 : day
# 8 : help
# 9 : die
# 10 : years
# 11 : better
# 12 : talk
# 13 : school
# 14 : someone
# 15 : every
# 16 : anyone
# 17 : fucking
# 18 : work
# 19 : need
# 20 : anymore