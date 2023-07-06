import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from spellchecker import SpellChecker

# Load the NRC EmoLex Lexicon
emolex_df = pd.read_csv('NRC-Emotion-Lexicon-Wordlevel-v0.92.csv')

# Convert the lexicon to a dictionary for efficiency
emolex_dict = emolex_df.pivot(index='word', 
                              columns='emotion', 
                              values='association').to_dict()

# Create lists of simple intensifiers and negations
intensifiers = ['very', 'extremely', 'absolutely', 'totally']
negations = ['not', 'never', 'no']

# Initialize spell checker
spell = SpellChecker()

# Now let's create a more complex sentiment analysis function
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
    
    return emotions

# Load your review data
reviews_df = pd.read_csv('reviews.csv')

# Create an empty DataFrame to store the sentiment for each sentence
sentences_df = pd.DataFrame(columns=['review', 'sentence', 'sentiment'])

# Tokenize each review into sentences and analyze sentiment
for index, row in reviews_df.iterrows():
    sentences = sent_tokenize(row['review'])
    for sentence in sentences:
        sentiment = sentiment_analysis(sentence)
        sentences_df = sentences_df.append({
            'review': row['review'],
            'sentence': sentence,
            'sentiment': sentiment
        }, ignore_index=True)

# Preview the data
print(sentences_df.head())
