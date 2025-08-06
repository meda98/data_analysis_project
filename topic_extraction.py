import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

def extract_topics_lsa(tfidf_df: pd.DataFrame, n_topics: int = 3, n_top_words: int = 10) -> None:
    """
    Performs Latent Semantic Analysis (LSA) on a TF-IDF matrix and prints top terms per topic.

    Args:
        tfidf_df (pd.DataFrame): DataFrame containing the TF-IDF feature matrix.
        n_topics (int): Number of topics to extract.
        n_top_words (int): Number of top words to display per topic.
    """
    terms = tfidf_df.columns
    X = tfidf_df.values

    lsa = TruncatedSVD(n_components=n_topics, random_state=42)
    lsa.fit(X)

    print("\nTop Terms per LSA Topic:\n")
    for i, comp in enumerate(lsa.components_):
        top_indices = np.argsort(comp)[::-1][:n_top_words]
        top_terms = [terms[j] for j in top_indices]
        print(f"Topic {i + 1}: {', '.join(top_terms)}")


def extract_topics_lda(bow_df: pd.DataFrame, n_topics: int = 3, n_top_words: int = 10) -> None:
    """
    Performs Latent Dirichlet Allocation (LDA) on a BoW matrix and prints top terms per topic.

    Args:
        bow_df (pd.DataFrame): DataFrame containing the BoW feature matrix.
        n_topics (int): Number of topics to extract.
        n_top_words (int): Number of top words to display per topic.
    """
    terms = bow_df.columns
    X = bow_df.values

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    print("\nTop Terms per LDA Topic:\n")
    for idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[::-1][:n_top_words]
        top_terms = [terms[i] for i in top_indices]
        print(f"Topic {idx + 1}: {', '.join(top_terms)}")