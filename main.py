if __name__ == "__main__":

    import pandas as pd
    from text_cleaning import clean_review_text
    from vectorization import vectorize_bow, vectorize_tfidf
    from topic_extraction import extract_topics_lsa, extract_topics_lda
    from coherence_score import determine_optimal_number_of_topics

    # Step 1: Load the 'ReviewBody' column from the CSV file
    AirlineReview_df = pd.read_csv('BA_AirlineReviews.csv', usecols=["ReviewBody"])

    # Step 2: Clean the reviews
    cleaned_AirlineReview_df = clean_review_text(AirlineReview_df)

    # Step 3: Generate BoW and TF-IDF matrices
    bow_df = vectorize_bow(cleaned_AirlineReview_df)            # for LDA
    tfidf_df = vectorize_tfidf(cleaned_AirlineReview_df)        # for LSA

    # Step 4: Determine best topic count for LSA and LDA using coherence
    best_n_topics_lsa = determine_optimal_number_of_topics(tfidf_df, cleaned_AirlineReview_df, model_type="lsa")
    best_n_topics_lda = determine_optimal_number_of_topics(bow_df, cleaned_AirlineReview_df, model_type="lda")

    # Step 5: Extract and display LSA and LDA topics
    extract_topics_lsa(tfidf_df, n_topics=best_n_topics_lsa)
    extract_topics_lda(bow_df, n_topics=best_n_topics_lda)