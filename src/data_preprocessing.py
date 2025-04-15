import nltk
import pandas as pd
import os.path as path
import os
import re
from nltk.corpus import stopwords

SETS_PATH = "data/data_preparation_phase"
nltk.download('stopwords')
nltk.download('wordnet')
#step 1: data exploration:
datasets = {'train':pd.read_csv(path.join(SETS_PATH,"train_set.csv")),
            'valid':pd.read_csv(path.join(SETS_PATH,"valid_set.csv")),
            'test':pd.read_csv(path.join(SETS_PATH,"test_set.csv"))}


#step 2: pre_processing pipeline:

def lower_case(text):
    return text.lower()

#why we did that? since emotions(happiness,sadness) are not usually correlated with this kind of emotions, likewise, symbols like !? could be used to represent angriness suspesion
def remove_puncitions(text):
    return re.sub(r'[^\w\s]',' ',text)

#numbers will not contribute to the emotions usually unless something like "I waited 30 years to get that. finally!"
def remove_numbers(text):
    return re.sub(r'[0-9]','',text)

#just did it eariler for tokenization later
def trim_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()


def remove_stopwords(text,stopwords=stopwords.words('english')):
    return " ".join(word for word in text.split(" ") if word not in stopwords)

def remove_links(text):
    return re.sub(r'\b(?:https?://|www\.)?\S+\.\S+\b', '', text)

def remove_repititve_patterns(text):
    # Step 1: Replace repeated characters (3+ times)
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # Step 2: Replace repeated words (2+ times)
    text = re.sub(r'\b(\w+)(?:\s+\1)+\b', r'\1', text)

    # Step 3: Replace repeated patterns (2+ times)
    text = re.sub(r'\b(.+?)\1+\b', r'\1', text)

    return text


def preprocessing(content,functions):
    for fun in functions:
        content = fun(content)
    return content


preprocessing_functions = [lower_case,remove_puncitions,remove_numbers,trim_extra_spaces,remove_stopwords,remove_links,remove_repititve_patterns,nltk.casual.casual_tokenize]


#let's tokenize all the sets and save them:
for s in datasets:
   datasets[s]['content'] = datasets[s]['content'].apply(lambda x: preprocessing(x,preprocessing_functions))


#let's check some samples of each and save results:d
for s in datasets:
    print(f"{s}_set:\n{datasets[s].head()}")
    datasets[s].to_csv(f"data/preprocessing_phase/{s}_set.csv")

