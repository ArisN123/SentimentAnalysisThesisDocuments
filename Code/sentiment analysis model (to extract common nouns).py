import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize, pos_tag
from spellchecker import SpellChecker
from collections import defaultdict

emolex_df = pd.read_csv('NRC-UPDATED.csv')
# Alter if necessary to add words that fit the specific context of the discussion

emolex_dict = emolex_df.pivot(index='word', 
                              columns='emotion', 
                              values='association').to_dict()

intensifiers = pd.read_csv('intensifiers.csv', header=None)[0].tolist()
mitigators = pd.read_csv('mitigators.csv', header=None)[0].tolist()

spell = SpellChecker()

def sentiment_analysis(text):
    emotions = {emotion: 0 for emotion in emolex_dict.keys()}
    
    words = word_tokenize(text.lower())

    for i, word in enumerate(words):
        corrected_word = spell.correction(word)
        
        if corrected_word in emolex_dict:
            intensity = 1
            
            for j in range(max(0, i-2), i):
                if words[j] in intensifiers:
                    intensity *= 2
                elif words[j] in negations:
                    intensity *= -1

            for emotion in emotions:
                emotions[emotion] += emolex_dict[corrected_word][emotion] * intensity
    
    positive = sum(emotions.values()) > 0

    return positive

reviews_df = pd.read_csv('AmazonReview.csv')
# Load the twitter data csv if you are trying to analyse tweets

noun_counter = defaultdict(lambda: defaultdict(int))

for index, row in reviews_df.iterrows():
    sentences = sent_tokenize(row['review'])
    for sentence in sentences:
        positive = sentiment_analysis(sentence)

        sentiment_label = 'positive' if positive else 'negative'

        tagged_words = pos_tag(word_tokenize(sentence))
        nouns = [word for word, pos in tagged_words if pos in ['NN', 'NNS']]

        for noun in nouns:
            noun_counter[noun][sentiment_label] += 1

results_df = pd.DataFrame(columns=['noun', 'count', 'sentiment'])

for noun, sentiment_counter in noun_counter.items():
    if sentiment_counter['positive'] > 0:
        results_df = results_df.append({
            'noun': noun,
            'count': sentiment_counter['positive'],
            'sentiment': 'positive'
        }, ignore_index=True)

    if sentiment_counter['negative'] > 0:
        results_df = results_df.append({
            'noun': noun,
            'count': sentiment_counter['negative'],
            'sentiment': 'negative'
        }, ignore_index=True)

results_df.to_csv('sentiment_results.csv', index=False)


