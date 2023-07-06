import pandas as pd
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from sklearn.metrics import confusion_matrix

# Load the NRC EmoLex Lexicon
emolex_df = pd.read_csv('NRC-UPDATED.csv')

# Create lists of intensifiers and negations
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

    # Calculate sentiment score as the sum of emotion scores normalized to range -1 to 1
    sentiment_score = sum(emotions.values())
    sentiment_score = -1 if sentiment_score < 0 else 1 if sentiment_score > 0 else 0

    return sentiment_score
 

# Load the test data
test_df = pd.read_csv('testtextdatafortheconfusionmatrix.csv')

# Apply the sentiment analysis model to the test data
test_df['predicted_sentiment'] = test_df['text'].apply(sentiment_analysis)

# Convert true sentiment from -1 to 1 scale to match the predicted sentiment
test_df['true_sentiment'] = test_df['true_sentiment'].apply(lambda x: -1 if x < 0 else 1 if x > 0 else 0)

# Generate the confusion matrix
cm = confusion_matrix(test_df['true_sentiment'], test_df['predicted_sentiment'])

# Print the confusion matrix
print(cm)
