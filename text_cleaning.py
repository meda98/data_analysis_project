import pandas as pd
import re
import os
import platform
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

# Get the path of the directory where the project is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Tell NLTK to look for its data files in the subfolder "nltk_data"
nltk.data.path.append(os.path.join(BASE_DIR, "nltk_data"))

# Initialize the standard NLTK stopword list for English
stop_words = set(stopwords.words("english"))

# Extend stopword list with generic and domain-specific terms
stop_words.update([
    # General-purpose stopwords
    'would', 'could', 'also', 'get', 'tell', 'day', 'minute',
    'hour', 'take', 'good', 'one',                                  
     # Airline-related terms
    'british', 'airways', 'airway', 'airline', 'flight', 'fly', 'air', 'plane',
    # Location/airport-specific terms
    'lhr', 'heathrow', 'gatwick', 'london'
])

# Initialize WordNet lemmatizer for converting words to base forms
lemmatizer = WordNetLemmatizer()

# Helper function to convert POS tags to WordNet format (needed for proper lemmatization)
def get_wordnet_pos(tag: str) -> str:
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default to noun

# Function to clean review text in a given DataFrame column
def clean_review_text(df: pd.DataFrame, text_column: str = "ReviewBody") -> pd.DataFrame:
    """
    Cleans text data in a specified column of a DataFrame using standard NLP preprocessing.

    This function performs the following steps on each text entry:
    - Converts to lowercase
    - Removes non-alphabetic characters
    - Tokenizes text into words
    - Applies POS tagging and lemmatization
    - Removes stopwords and very short words
    - Rejoins the tokens into a cleaned string

    Args:
        df (pd.DataFrame): The DataFrame containing the text data.
        text_column (str, optional): The name of the column to clean. Defaults to "ReviewBody".

    Returns:
        pd.DataFrame: A new DataFrame with the specified column cleaned in place.
    """


    # Main text cleaning function applied to each row
    def clean_text(text: str) -> str:
        if pd.isnull(text):
            return ""

        # Lowercase the text
        text = text.lower()

        # Remove all characters except letters and whitespace
        text = re.sub(r"[^a-z\s]", " ", text)

        # Tokenize the cleaned string into words
        tokens = word_tokenize(text)

        # POS tagging for better lemmatization
        tagged = pos_tag(tokens)

        # Lemmatize each word based on its POS tag
        lemmatized = [
            lemmatizer.lemmatize(word, get_wordnet_pos(pos))
            for word, pos in tagged
        ]

        # Remove stopwords and very short words after lemmatization
        filtered = [
            word for word in lemmatized
            if word not in stop_words and len(word) > 2
        ]

        # Rejoin words into a cleaned single string
        return " ".join(filtered).strip()
    
    # Apply cleaning to each review in the specified column
    df[text_column] = df[text_column].apply(clean_text)
    
    return df