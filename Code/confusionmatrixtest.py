import pandas as pd
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from sklearn.metrics import confusion_matrix

emolex_df = pd.read_csv('NRC-UPDATED.csv')

intensifiers = [
    'very', 'extremely', 'absolutely', 'totally', 'incredibly', 'rather', 'quite', 'highly', 
    'exceptionally', 'especially', 'particularly', 'super', 'seriously', 'remarkably', 
    'majorly', 'most', 'more', 'so', 'really'
]

negations = [
    'not', 'never', 'no', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'never',
    'hardly', 'scarcely', 'barely', 'doesn’t', 'isn’t', 'wasn’t', 'shouldn’t', 'wouldn’t', 
    'couldn’t', 'won’t', 'can’t', 'don’t'
]

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
 
test_df = pd.read_csv('testtextdatafortheconfusionmatrix.csv')

test_df['predicted_sentiment'] = test_df['text'].apply(sentiment_analysis)

test_df['true_sentiment'] = test_df['true_sentiment'].apply(lambda x: -1 if x < 0 else 1 if x > 0 else 0)

cm = confusion_matrix(test_df['true_sentiment'], test_df['predicted_sentiment'])

print(cm)
