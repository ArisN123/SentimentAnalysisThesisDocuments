import pandas as pd
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

emolex_df = pd.read_csv('NRC-UPDATED.csv')

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

    sentiment_score = sum(emotions.values())
    sentiment_score = -1 if sentiment_score < 0 else 1 if sentiment_score > 0 else 0

    return sentiment_score

# Switch csv file with AmazonReviews.csv when analysing the Amazon data
df = pd.read_csv('McDoTwitterdata.csv')

df['sentiment_score'] = df['text_column'].apply(sentiment_analysis)

df.to_csv('reviews_with_sentiment.csv', index=False)
