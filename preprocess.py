import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.casual import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag
from urllib.parse import urlsplit
import re

# initialise
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
tokenizer = TweetTokenizer(preserve_case=False)

#HappyEmoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])

emoticons = emoticons_happy.union(emoticons_sad)    

def process_one_tweet(tweet):

    new_tweet_ls = []

    for word in tokenizer.tokenize(tweet):

        word = re.sub("\'", "", word).lower()
        
        if len(word) <= 1 or word == "rt":
            continue  # ignore single chars and rt

        if word in emoticons:
            new_tweet_ls.append(word)   # keep emoticons
        elif word[0] == "#" or word[0] == "@":
            new_tweet_ls.append(word)   # keep hashtags
        elif word.startswith("http"):   # keep url
            base_url = "{0.scheme}://{0.netloc}/".format(urlsplit(word))
            new_tweet_ls.append(base_url)
        elif word.startswith("www."):   # keep url
            base_url = "{0.scheme}://{0.netloc}/".format(urlsplit("http://" + word))
            new_tweet_ls.append(base_url)
        elif re.search("[0-9]", word):  # parase numbers into special token: #NUM
            new_tweet_ls.append("#NUM")
        elif not (word in stop_words or re.search("\W",word)):  # regular words
                new_tweet_ls.append(lemmatizer.lemmatize(word))

    return new_tweet_ls

def remove_test_users(tweet, all_users):
    
    # lower case
    all_users = [user.lower() for user in all_users]
    new_tweet_ls = []

    for word in tokenizer.tokenize(tweet):
        if word[0] != "@":
            new_tweet_ls.append(word)
        elif word[1:] in all_users: # only keep relevant users
            new_tweet_ls.append(word)
    
    return " ".join(new_tweet_ls)