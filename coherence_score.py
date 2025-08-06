from typing import Literal
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

def determine_optimal_number_of_topics(
    feature_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    model_type: Literal["lsa", "lda"],
    text_column: str = "ReviewBody",
    min_topics: int = 2,
    max_topics: int = 10,
    step: int = 1,
    top_n: int = 10
) -> int:
    """
    Generic function to compute coherence scores for LSA or LDA models and return
    the optimal number of topics.

    Args:
        feature_df (pd.DataFrame): TF-IDF or BoW matrix (documents x terms).
        cleaned_df (pd.DataFrame): DataFrame with cleaned review text.
        model_type (Literal["lsa", "lda"]): Choose between 'lsa' or 'lda'.
        text_column (str): Column containing cleaned review text.
        min_topics (int): Minimum number of topics to test.
        max_topics (int): Maximum number of topics to test.
        step (int): Step size for topic range.
        top_n (int): Number of top terms per topic to use in coherence scoring.

    Returns:
        int: Optimal number of topics with the highest coherence score.
    """
    # Tokenize text for coherence model
    tokenized_texts = cleaned_df[text_column].apply(str.split).tolist()

    # Gensim dictionary
    dictionary = Dictionary(tokenized_texts)

    # Matrix and terms from vectorized input
    X = feature_df.values
    terms = feature_df.columns

    best_score = -1
    best_n_topics = None

    print(f"\n{model_type.upper()} Coherence Scores:\n")

    for n_topics in range(min_topics, max_topics + 1, step):
        if model_type == "lsa":
            model = TruncatedSVD(n_components=n_topics, random_state=42)
            model.fit(X)
            components = model.components_

        elif model_type == "lda":
            model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                learning_method='batch',
                max_iter=10
            )
            model.fit(X)
            components = model.components_

        else:
            raise ValueError("Invalid model_type. Use 'lsa' or 'lda'.")

        # Extract top terms for each topic
        topics = []
        for comp in components:
            top_indices = comp.argsort()[::-1][:top_n]
            topic_words = [terms[i] for i in top_indices]
            topics.append(topic_words)

        # Compute coherence
        coherence_model = CoherenceModel(
            topics=topics,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        score = coherence_model.get_coherence()
        print(f"Topics: {n_topics} | Coherence Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_n_topics = n_topics

    print(f"\nBest {model_type.upper()} topic count: {best_n_topics} | Coherence Score: {best_score:.4f}\n")
    return best_n_topics