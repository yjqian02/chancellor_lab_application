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
from collections import defaultdict
import nltk
# nltk.download("stopwords")
from nltk.corpus import stopwords


# global lists for part 5
# altered list from stopwords.words() from nltk 
# "" is to account for first index
common = ['i', 'im', 'me', 'my', 'myself', 'we', 'our', 'ours',
    'ourselves', 'you', "youre", "yove", "youll", "youd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
    'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
    'it', "its", 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "tha'll", 'these', 'those', 'am', 'is',
    'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o',
    're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didnt", 'doesn', "doesnt", 'hadn',
    "hadn't", 'hasn', "hasnt", 'haven', "havent", 'isn', "isnt", 'ma', 'mightn', "mightnt", 'mustn',
    "mustn't", 'needn', "neednt", 'shan', "shant", 'shouldn', "shouldn't", 'wasn', "wasnt", 'weren',
    "werent", 'won', "wont", 'wouldn', "wouldnt", "dont", "ive", "like", "know", "even", "one", "would", "much", 
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
    word_prob = defaultdict(int)
    sum = 0
    for i in range(len(posts_clean)):
        post = str(posts_clean[i])
        post = post.split(" ")
        if post != "[deleted]" and post != "[removed]":
            sum += len(post)
            for word in post:
                word_prob[word] += 1
                if word in d_vocab:
                        word_prob[word] += 1 # increase probabiity of risk words
            
    # date range
    dates = data["created_utc"]
    dt_dates = []
    for i in range(len(dates)):
        if "http" not in dates[i]: # skip over invalid dates/erroring data
            dt_dates.append(datetime.fromtimestamp(int(dates[i])))
    dt_dates.sort()

    # 20 most important posts
    top_twenty = {}
    count = 1
    word_prob = dict(sorted(word_prob.items(), key=lambda x:x[1]))
    for i, (word,prob) in enumerate(reversed(word_prob.items())):
        if count < 21 and word.lower() not in common:
            top_twenty[count] = word
            count += 1

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


# PS C:\Users\a1z26\Documents\chancellor_lab_application> python .\tabulation.py
# C:\Users\a1z26\Documents\chancellor_lab_application\tabulation.py:85: FutureWarning: The default value of regex will change from True to False in a future version.
#   posts_clean = data["selftext"].str.replace(r'[^\w\s]', '', flags=re.UNICODE)
# There are 30000 posts in the dataset.

# There are 342 deleted posts in the dataset.      

# There are 3082 removed posts in the dataset.     

# There are 24725 unique authors in the dataset.   

# The average post length is 172.75 words.

# The date range of the datset is:
# 2019-07-27 20:33:34 to 2020-11-23 20:30:46       

# The twenty most important words in the posts are:
# 1 : time
# 2 : feel
# 3 : friends
# 4 : want
# 5 : depression
# 6 : life
# 7 : help
# 8 : cant
# 9 : family
# 10 : talk
# 11 : nothing
# 12 : depressed
# 13 : need
# 14 : never
# 15 : think
# 16 : day
# 17 : alone
# 18 : parents
# 19 : years
# 20 : tired