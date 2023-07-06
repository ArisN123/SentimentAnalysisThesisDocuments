import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize, pos_tag
from spellchecker import SpellChecker
from collections import defaultdict

# Load the NRC EmoLex Lexicon
emolex_df = pd.read_csv('NRC-UPDATED.csv')
# Alter if necessary to add words that fit the specific context of the discussion

# Convert the lexicon to a dictionary for efficiency
emolex_dict = emolex_df.pivot(index='word', 
                              columns='emotion', 
                              values='association').to_dict()

# Create lists of simple intensifiers and negations
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

# Initialize spell checker
spell = SpellChecker()

# Begin sentiment analysis process
def sentiment_analysis(text):
    # Initialize scores for each emotion to 0
    emotions = {emotion: 0 for emotion in emolex_dict.keys()}
    
    # Tokenize the text into words
    words = word_tokenize(text.lower())

    # For each word, check for intensifiers or negations before it, and add the value it represents for each emotion
    for i, word in enumerate(words):
        corrected_word = spell.correction(word)
        
        if corrected_word in emolex_dict:
            # Initialize intensity
            intensity = 1
            
            # Check for intensifier and/or negation up to two words before the word
            for j in range(max(0, i-2), i):
                if words[j] in intensifiers:
                    intensity *= 2
                elif words[j] in negations:
                    intensity *= -1

            for emotion in emotions:
                emotions[emotion] += emolex_dict[corrected_word][emotion] * intensity
    
    # Determine whether the sentiment is positive
    positive = sum(emotions.values()) > 0

    return positive

# Load the Amazon review data
# Load the twitter data csv if you are trying to analyse those
reviews_df = pd.read_csv('AmazonReview.csv')

# Initialize a defaultdict to count nouns
noun_counter = defaultdict(lambda: defaultdict(int))

# Tokenize each review into sentences, analyze sentiment, and identify nouns
for index, row in reviews_df.iterrows():
    sentences = sent_tokenize(row['review'])
    for sentence in sentences:
        positive = sentiment_analysis(sentence)

        # Determine the sentiment label
        sentiment_label = 'positive' if positive else 'negative'

        tagged_words = pos_tag(word_tokenize(sentence))
        nouns = [word for word, pos in tagged_words if pos in ['NN', 'NNS']]

        for noun in nouns:
            noun_counter[noun][sentiment_label] += 1

# Create a DataFrame to store the results
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

# Export to CSV
results_df.to_csv('sentiment_results.csv', index=False)


