#### Import Packages ####
import boto3
import uuid
import time
from datetime import datetime
from decimal import *
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re
import string
from lyricsgenius import Genius

#### Variables ####
ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])

GENIUS_API_TOKEN = "ohCF3KM7xRoZi5EAfx5wNPwFDRIYrIog80ipQdzFXOkWaU4wre04NeTks1kB50Cy"

#### Functions ####
def get_lyrics(song, artist, token):
    genius = Genius(token)
    try:
        return genius.search_song(song, artist).lyrics
    except AttributeError:
        return ''
    except:
        time.sleep(5)
        return genius.search_song(song, artist).lyrics
        
    

def filelist(root):
    """Return a fully-qualified list of filenames under root directory"""
    allfiles = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            allfiles.append(os.path.join(path, name))
    return allfiles

def load_glove(filename):
    """
    Read all lines from the indicated file and return a dictionary
    mapping word:vector where vectors are of numpy `array` type.
    GloVe file lines are of the form:

    the 0.418 0.24968 -0.41242 0.1217 ...

    So split each line on spaces into a list; the first element is the word
    and the remaining elements represent factor components. The length of the vector
    should not matter; read vectors of any length.

    When computing the vector for each document, use just the text, not the text and title.
    """
    word2vec = {}
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split(' ')
            word2vec[line[0]] = np.asarray(line[1:], "float32")
    return word2vec

def words(text):
    """
    Given a string, return a list of words normalized as follows.
    Split the string to make words first by using regex compile() function
    and string.punctuation + '0-9\\r\\t\\n]' to replace all those
    char with a space character.
    Split on space to get word list.
    Ignore words < 3 char long.
    Lowercase all words
    Remove English stop words
    """
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text)  # delete stuff but leave at least a space to avoid clumping together
    words = nopunct.split(" ")
    words = [w.lower() for w in words]
    words = [w for w in words if len(w) > 2]
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]  # ignore a, an, to, at, be, ...
    return words

def doc2vec(text, gloves, tokenizer=None):
    """
    Return the word vector centroid for the text. Sum the word vectors
    for each word and then divide by the number of words. Ignore words
    not in gloves.
    """
    if tokenizer:
        vec = [gloves[w] for w in tokenizer.tokenize(text) if w in gloves]
    else:
        vec = [gloves[w] for w in words(text) if w in gloves]
    total = np.sum(vec, axis = 0)
    return total / len(vec)

def get_users_to_posts(aws_access_key, aws_secret_key):
    """
    Connect to database where Ay-Yo! posts are stored to 
    create training data.
    
    Parameters
    ----------
    aws_access_key: str
    aws_secret_key: str
        Keys necessary to access Ay-Yo's AWS DynamoDB
    
    Returns
    -------
    users_to_posts: dict
        keys are unqiue user ids
        values are list of posts made by that user
        {user_id: [
                    (song_name, song_artist),
                    (song_name2, song_artist2),
                    ...
                    ]
        }

    """
    # connect to dynamo
    dynamodb = boto3.resource(
                    'dynamodb',
                    region_name='us-west-1',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key)

    # scan all posts and make dataframe
    post_table = dynamodb.Table('posts')
    post_df = pd.DataFrame(post_table.scan()['Items'])

    # create dictionary
    user_to_posts = {u: [] for u in np.unique(post_df['posted_by'])}
    for _, row in post_df.iterrows():
        user_to_posts[row['posted_by']].append((row['song_name'], row['artist']))
    
    return user_to_posts