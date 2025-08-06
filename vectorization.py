import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def vectorize_bow(df: pd.DataFrame, text_column: str = "ReviewBody", max_features: int = 5000) -> pd.DataFrame:
    """
    Generates a Bag-of-Words (BoW) representation with unigrams and bigrams from the specified text column.

    Args:
        df (pd.DataFrame): DataFrame containing the cleaned text.
        text_column (str): Name of the column with text data to vectorize.
        max_features (int): Maximum number of features to include in the matrix.

    Returns:
        pd.DataFrame: A DataFrame containing the BoW feature matrix.
    """
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(df[text_column])
    bow_df = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return bow_df


def vectorize_tfidf(df: pd.DataFrame, text_column: str = "ReviewBody", max_features: int = 5000) -> pd.DataFrame:
    """
    Generates a TF-IDF representation with unigrams and bigrams from the specified text column.

    Args:
        df (pd.DataFrame): DataFrame containing the cleaned text.
        text_column (str): Name of the column with text data to vectorize.
        max_features (int): Maximum number of features to include in the matrix.

    Returns:
        pd.DataFrame: A DataFrame containing the TF-IDF feature matrix.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(df[text_column])
    tfidf_df = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return tfidf_df