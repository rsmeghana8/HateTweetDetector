import pandas as pd
import numpy as np
import streamlit as st
import nltk

import re
import warnings
warnings.filterwarnings('ignore')
from nltk.stem import WordNetLemmatizer
lemm= WordNetLemmatizer()
nltk.download('wordnet')
import pickle
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
from sklearn.preprocessing import normalize
from scipy.sparse import hstack



test_tweet = 'Have a nice day #happy'
tweet = st.text_input('Input a tweet here', test_tweet)

button = st.button("Click to find the sentiment of your tweet")
if button:
    if tweet != '' and tweet != '#':
        with st.spinner(f'Analyzing ..'):
            df = pd.DataFrame({u"tweet": [tweet]})



            def emojis(text):
                emoji = re.compile("["u"\U0001F600-\U0001F64F"  # emoticons
                                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                    u"\U00002702-\U000027B0"
                                    u"\U000024C2-\U0001F251"
                                    "]+", flags=re.UNICODE)
                return emoji.sub(r'', text)

            def len_tweet(tweet):
                x = re.sub(r'#(\w+)', '', tweet).strip()
                y = re.sub('[^A-Za-z0-9]+',' ',x).strip()
                return len(y)

            contractions = {"ain't": "am not","aren't": "are not ","can't": "cannot","can't've": "cannot have",
            "'cause": "because","could've": "could have",
            "couldn't": "could not","couldn't've": "could not have",
            "didn't": "did not","doesn't": "does not","don't": "do not",
            "hadn't": "had not","hadn't've": "had not have",
            "hasn't": "has not","haven't": "have not",
            "he'd": "he would","he'd've": "he would have",
            "he'll": "he will","he'll've": "he will have",
            "he's": "he is","how'd": "how did","how'd'y": "how do you","how'll": "how will",
            "how's": "how has / how is / how does","i'd": "i would",
            "i'd've": "i would have","i'll": "I will","i'll've": "i will have","i'm": "i am",
            "i've": "i have","isn't": "is not","it'd": "it would","it'd've": "it would have",
            "it'll": "it will","it'll've": "it will have","it's": "it is","let's": "let us",
            "ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not",
            "mightn't've": "might not have","must've": "must have","mustn't": "must not","mustn't've": "must not have",
            "needn't": "need not","needn't've": "need not have",
            "o'clock": "of the clock","oughtn't": "ought not","oughtn't've": "ought not have",
            "shan't": "shall not","sha'n't": "shall not",
            "shan't've": "shall not have", "she'd": "she would","she'd've": "she would have",
            "she'll": "she will","she'll've": "she will have",
            "she's": "she is","should've": "should have","shouldn't": "should not",
            "shouldn't've": "should not have","so've": "so have","so's": "so as","that'd": "that would","that'd've": "that would have",
            "that's": "that is","there'd": "there would","there'd've": "there would have",  "there's": "there is",
            "they'd": "they would","they'd've": "they would have","they'll": "they will","they'll've": "they will have",
            "they're": "they are","they've": "they have","to've": "to have","wasn't": "was not","we'd": "we would",
            "we'd've": "we would have","we'll": "we will","we'll've": "we will have","we're": "we are","we've": "we have","weren't": "were not",
            "what'll": "what will","what'll've": "what will have",
            "what're": "what are","what's": "what is","what've": "what have",
            "when's": "when is","when've": "when have","where'd": "where did","where's": "where is",
            "where've": "where have","who'll": "who will","who'll've": "who will have","who's": "who is",
            "who've": "who have","why's": "why is","why've": "why have","will've": "will have",
            "won't": "will not","won't've": "will not have","would've": "would have",
            "wouldn't": "would not","wouldn't've": "would not have","y'all": "you all",
            "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would","you'd've": "you would have",
            "you'll": "you will","you'll've": "you will have","you're": "you are","you've": "you have",
            'gr8':'great',

            }

            # Stop words based on the train data
            stopwords = {'u','user','from', 'its', 'therefore',
            "they", "were", "youre", "wont", 'who', 'you', 
            "they", 'some', 'their', 'ours', "well", 'that',
            'this', 'most', 'once', "she", 'is', 'com', 'his',
            'ought', "have", 'myself', 'on', 'own', 'your', "im",
            'which', 'after', 'further', 'shall', 'other', 'been', 'nor',
            'under', 'against', "did", 'itself',
            'where', 'at', 'them', "i", 'but', "does",
            'a', 'could', "was", 'while', 
            'yourselves', 'our', 'would', "let", 
            'can', 'are', 'here', 'how', 'through', 
            'had', 'over', "that", 'to', 'be', 'why', 
            "he", 'same', 'than', 'above', 'these', 
            'were', 'hence', 'k', 'more', 'they', "hadn't",
            'by', "they", 'hers', "are", "must", 
            'yours', 'himself', 'otherwise', 
            'in', 'down', 'as', 'whom', 'doing', 'does',
            'me', 'or', "she", 'cannot', 'have', "can't", 
            'we', 'with', "how", 'having', 'since',
            'of', 'www', 'for', 'should', 'up', 'yourself',
            'those', "where", 'http', 'about', 'my', 'very', 
            'again', 'and', "why", 'was', 
            'any', "should", 'her', 'below', 'get',
            'just', 'also', 'each', 'herself', "there", 'what',
            'so', "would", "we'd", 'during', "they", 'such', 
            'am', "hasn't",  'until', "you've", 'then', "here", 
            "when", 'he', 'both', 'i',  'an', 'the', 'into', "we", 
            'else', 'did', 'because', 'when', "you",
            'too', 'between', 'r', 'do',
            "could", 'has', 'off', 'theirs', 'ever',
            'there', 'him', "i", 'all', 'it', 'before'}

            def decontracted(phrase):
                for key in contractions.keys():
                    phrase = re.sub(key, contractions.get(key), phrase)
                return phrase

            def remove_digits(s):
                no_digits = ''
                for w in s.split():
                    result = ''.join([i for i in w if  not i.isdigit()])
                    no_digits += result + " "
                return no_digits

            def remove_pattern(text): # to remove patterns like @user
                r = re.findall('@[\w]*', text)
                for i in r:
                    text = re.sub(i,'',text)
                return text

            def preprocessing (df, tweet):
                
                df[tweet] = df[tweet].apply(remove_pattern)
                df[tweet]= df[tweet].str.lower()

                # Removing Html charecters if any
                df[tweet]=df[tweet].str.replace("<[^<]+?>","",regex=True) 
                    
                # Cleaning the emojis if any
                df[tweet]=df[tweet].apply(lambda x: emojis(x))
                
                # Creating a new col. with no of hashtags
                df['no_of_hashtags'] = df[tweet].apply(lambda x: len(re.findall(r'#(\w+)',x)))  
                
                # Length of tweet without hashtag
                df['tweet_len'] = df[tweet].apply(len_tweet)
                
                df[tweet] = df[tweet].apply(decontracted)
                df[tweet] = df[tweet].apply(lambda x :re.sub('[^A-Za-z0-9]+',' ',x).strip())
                df[tweet] =  df[tweet].apply(lambda x: " ".join(lemm.lemmatize(word) for word in x.split()))
                df[tweet] =  df[tweet].apply(remove_digits)
                df[tweet] = df[tweet].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
                df[tweet] = df[tweet].apply(lambda x: ' '.join([w for w in x.split() if w not in stopwords]))
            
            preprocessing(df,'tweet')

            with open('bow.pkl','rb') as f:
                vectorizer = pickle.load(f)

            with open('lr_model.pkl','rb') as f:
                model = pickle.load(f)

            with open('norms.pkl', 'rb') as f:  
                norm_hs, norm_len = pickle.load(f)

            v = []
            for i in df['tweet']:
                snt = sid.polarity_scores(i)
                v.append(snt)
            data_fr_te = pd.DataFrame(v)
            x_test_senti = np.array(data_fr_te)

            x_test_bow = vectorizer.transform(df['tweet'].values)

            x_test_hs = (df['no_of_hashtags'].values.reshape(1,-1)/norm_hs).T
            x_test_len = (df['tweet_len'].values.reshape(1,-1)/norm_len).T 

            x_test = hstack((x_test_bow,
                            x_test_hs,
                            x_test_len,
                        x_test_senti)).tocsr()

            clss = model.predict(x_test)
            prob = model.predict_proba(x_test)
            a = np.argmax(prob)
            if a == 0:
                st.write('{} % sure that the tweet is not hateful/racist/sexist '.format(round (prob[0][0] * 100),2))
            else: 
                st.write('{} % sure that the tweet is hateful/racist/sexist'.format(round (prob[0][1] * 100),2))
            
