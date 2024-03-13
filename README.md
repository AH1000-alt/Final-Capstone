# Final-Capstone
## 1. A description of the dataset used.
The dataset used for this sentiment analysis is a collection of product reviews from Amazon.
The dataset contains over 1 million reviews of various products, including electronics, books,
clothing, and home goods. The dataset used is very informative as it shows a wide range of reviews
for the same product which allows for easier, further research.
## 2. Details of the preprocessing steps.
The text data was preprocessed by removing non-alphabet characters, converting to lowercase, and
trimming whitespaces.
The text data in the 'reviews.text' column is converted to lowercase and
whitespace is stripped to ensure consistency. Used .str.lower() and .str.strip().
The resulting preprocessed text is stored in a new column named
'cleaned_reviews'.
## 3. Evaluation of results.
The sentiment analysis model predictedsentiment scores for the product reviews, with sample
reviews and similarity scores provided.
Sentiment analysis is performed on the preprocessed text using the
spaCyTextBlob extension, which provides polarity scores for each review.
The polarity score indicates the sentiment of each review, with positive values
representing positive sentiment, negative values representing negative
sentiment, and zero representing neutral sentiment.
The similarity between pairs of reviews is computed using spaCy's similarity
function, which compares the semantic similarity between the preprocessed
documents.
### 4. Insights into the model's strengths and limitations.
The model demonstrates accuracy in predicting sentiment based on the text data. However, further
analysis is needed to evaluate its performance on a larger dataset.
### Strengths:
Utilises the spaCyTextBlob extension, which combines the power of spaCy for
natural language processing with TextBlob for sentiment analysis.
Provides a quick and easy way to preprocess text data and perform sentiment
analysis without the need for extensive feature engineering.
Incorporates error handling to handle cases where sentiment analysis cannot
be performed, ensuring robustness.
### Limitations:
Preprocessing steps such as stopword removal and lemmatisation may
oversimplify the text data, potentially losing important context or nuance.
The accuracy of the sentiment analysis and similarity calculation heavily
depends on the quality of the preprocessed text and the capabilities of the
underlying spaCy models.
