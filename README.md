# TuneType
Predicting Song Genres using Audio and Lyrics

This project builds and compares machine learning models to classify songs into ten major music genres using a large Spotify dataset containing audio features and song lyrics. An audio-only baseline model using Spotify’s engineered acoustic features established initial performance and revealed strong separability for some genres (such as Classical and Hip-Hop) alongside substantial overlap among others (notably Rock and Pop). A lyrics-only model based on TF-IDF representations captured genre-specific linguistic patterns and improved performance for language-driven genres, while highlighting limitations for genres with sparse or theatrical lyrics.

A multimodal model combining scaled audio features with TF-IDF lyric representations achieved the best overall performance, demonstrating that acoustic and linguistic information provide complementary signals for genre classification. Hyperparameters were tuned using stratified cross-validation on a representative subset of the data to balance computational efficiency and performance. Model interpretability analyses examined audio feature coefficients, top genre-specific lyric terms, and confusion matrices, revealing meaningful genre characteristics and consistent ambiguities between closely related styles. This project demonstrates an end-to-end machine learning workflow on large real-world data, emphasizing multimodal learning, interpretability, and principled model evaluation.

URL for Streamlit App: https://spotify-genre-classification-dghzck2kg45oj3necqpsxi.streamlit.app/ 

Original Dataset: https://www.kaggle.com/datasets/serkantysz/550k-spotify-songs-audio-lyrics-and-genres
