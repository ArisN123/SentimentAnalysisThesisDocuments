import pandas as pd
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker

# Load the NRC EmoLex Lexicon
emolex_df = pd.read_csv('NRC-UPDATED.csv')

# Convert the lexicon to a dictionary for efficiency
emolex_dict = emolex_df.pivot(index='word', 
                              columns='emotion', 
                              values='association').to_dict()

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

# Begin sentiment analysis process
def sentiment_analysis(text):
    # Initialize scores for each emotion to 0
    emotions = {emotion: 0 for emotion in emolex_dict.keys()}
    
    # Tokenize the text into words
    words = word_tokenize(text.lower())

    # For each word, check for intensifiers or negations before it (2-word range), and add the value it represents for each emotion
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

# Load Twitter data for analysis
# Switch csv file with AmazonReviews.csv when analysing the Amazon data
df = pd.read_csv('McDoTwitterdata.csv')

# Apply sentiment analysis to each row and store the results in a new column
df['sentiment_score'] = df['text_column'].apply(sentiment_analysis)

# Save the DataFrame with the sentiment scores back to a csv file
df.to_csv('reviews_with_sentiment.csv', index=False)
