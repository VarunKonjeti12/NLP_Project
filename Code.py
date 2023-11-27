#METHOD 1
### PREPROCESSING (3-81)
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from transformers import BertTokenizer # If we are using BERT Model
from sklearn.model_selection import train_test_split

# Sample dataset
reviews = [
    {"text": "The camera is great, but the battery life is disappointing.", "aspect_sentiment": {"camera": "positive", "battery": "negative"}},
    {"text": "Easy to use and affordable.", "aspect_sentiment": {"usability": "positive", "price": "positive"}},
    # Add more reviews to the dataset
]

# Preprocessing steps
cleaned_reviews = []
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
contractions = {"don't": "do not", "can't": "cannot", "won't": "will not"}

for review in reviews:
    # Step 1: Text Cleaning
    cleaned_text = re.sub(r'<.*?>', '', review["text"])
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', cleaned_text)
    cleaned_text = cleaned_text.lower()

    # Step 2: Tokenization
    tokens = word_tokenize(cleaned_text)

    # Step 3: Stopword Removal
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Step 4: Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Step 7: Part-of-Speech (POS) Tagging
    pos_tags = pos_tag(lemmatized_tokens)

    # Step 8: Named Entity Recognition (NER)
    ner_tags = ne_chunk(pos_tags)

    # Step 9: Remove POS and NER tags
    lemmatized_tokens = [token for token, pos in pos_tags]

    # Step 10: Remove Non-Aspect Words
    non_aspect_words = ["generic_word1", "generic_word2"]
    lemmatized_tokens = [token for token in lemmatized_tokens if token not in non_aspect_words]

    # Step 11: Handling Negations
    for i, token in enumerate(lemmatized_tokens):
        if token == "not" and i < len(lemmatized_tokens) - 1:
            lemmatized_tokens[i + 1] = "not_" + lemmatized_tokens[i + 1]

    # Step 12: Handling Contractions
    lemmatized_tokens = [contractions.get(token, token) for token in lemmatized_tokens]

    # Step 13: Create Contextual Embeddings (using BERT tokenizer)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    contextual_embeddings = tokenizer.encode(cleaned_text, add_special_tokens=True)

    # Update the review dictionary
    review["cleaned_text"] = cleaned_text
    review["lemmatized_tokens"] = lemmatized_tokens
    review["contextual_embeddings"] = contextual_embeddings

    cleaned_reviews.append(review)

# Step 14: Data Splitting
train_data, test_data = train_test_split(cleaned_reviews, test_size=0.2, random_state=42)

# Display the preprocessed data
for idx, review in enumerate(train_data):
    print(f"Review {idx + 1}:")
    print(f"  Cleaned Text: {review['cleaned_text']}")
    print(f"  Lemmatized Tokens: {review['lemmatized_tokens']}")
    print(f"  Contextual Embeddings: {review['contextual_embeddings']}")
    print(f"  Aspect Sentiment: {review['aspect_sentiment']}")
    print("\n")

# METHOD 2
###PREPROCESSING (84-241)
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Sample dataset
reviews = [
    {"text": "The camera is great, but the battery life is disappointing.", "aspect_sentiment": {"camera": "positive", "battery": "negative"}},
    {"text": "Easy to use and affordable.", "aspect_sentiment": {"usability": "positive", "price": "positive"}},
    # Add more reviews to the dataset
]

# Step 1: Text Cleaning
cleaned_reviews = []
for review in reviews:
    cleaned_text = re.sub(r'<.*?>', '', review["text"])  # Remove HTML tags
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', cleaned_text)  # Remove special characters and numbers
    cleaned_text = cleaned_text.lower()  # Convert to lowercase
    review["cleaned_text"] = cleaned_text
    cleaned_reviews.append(review)

# Step 2: Tokenization
tokenized_reviews = []
for review in cleaned_reviews:
    tokens = word_tokenize(review["cleaned_text"])
    review["tokens"] = tokens
    tokenized_reviews.append(review)

# Step 3: Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_reviews = []
for review in tokenized_reviews:
    filtered_tokens = [token for token in review["tokens"] if token not in stop_words]
    review["filtered_tokens"] = filtered_tokens
    filtered_reviews.append(review)

# Step 4: Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_reviews = []
for review in filtered_reviews:
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in review["filtered_tokens"]]
    review["lemmatized_tokens"] = lemmatized_tokens
    lemmatized_reviews.append(review)

# Display the preprocessed data
for idx, review in enumerate(lemmatized_reviews):
    print(f"Review {idx + 1}:")
    print(f"  Original Text: {review['text']}")
    print(f"  Cleaned Text: {review['cleaned_text']}")
    print(f"  Tokens: {review['tokens']}")
    print(f"  Filtered Tokens: {review['filtered_tokens']}")
    print(f"  Lemmatized Tokens: {review['lemmatized_tokens']}")
    print(f"  Aspect Sentiment: {review['aspect_sentiment']}")
    print("\n")

# step 5 aspect based extraction
aspect_keywords = ["camera", "battery", "usability", "price"]  # Define a list of relevant aspects

# Extract aspects from lemmatized tokens
for review in lemmatized_reviews:
    extracted_aspects = [aspect for aspect in aspect_keywords if aspect in review["lemmatized_tokens"]]
    review["extracted_aspects"] = extracted_aspects

# Display the extracted aspects for each review
for idx, review in enumerate(lemmatized_reviews):
    print(f"Review {idx + 1}:")
    print(f"  Extracted Aspects: {review['extracted_aspects']}")
    print("\n")
  
# step 6 feature engineering
# Create a bag-of-words representation for each review
for review in lemmatized_reviews:
    review["bag_of_words"] = {token: True for token in review["lemmatized_tokens"]}

# Display the bag-of-words representation for each review
for idx, review in enumerate(lemmatized_reviews):
    print(f"Review {idx + 1}:")
    print(f"  Bag-of-Words Representation: {review['bag_of_words']}")
    print("\n")
  
# step 7 pos tagging
from nltk import pos_tag
# Perform POS tagging on lemmatized tokens
pos_tagged_reviews = []
for review in lemmatized_reviews:
    pos_tags = pos_tag(review["lemmatized_tokens"])
    review["pos_tags"] = pos_tags
    pos_tagged_reviews.append(review)

# Display POS tags for each review
for idx, review in enumerate(pos_tagged_reviews):
    print(f"Review {idx + 1}:")
    print(f"  POS Tags: {review['pos_tags']}")
    print("\n")

# step 8 NER
from nltk import ne_chunk

# Perform NER on POS-tagged tokens
ner_reviews = []
for review in pos_tagged_reviews:
    ner_tags = ne_chunk(review["pos_tags"])
    review["ner_tags"] = ner_tags
    ner_reviews.append(review)

# Display NER tags for each review
for idx, review in enumerate(ner_reviews):
    print(f"Review {idx + 1}:")
    print(f"  NER Tags: {review['ner_tags']}")
    print("\n")

# step 9 Remove POS Tags and NER Tags
# Remove POS and NER tags to keep only the lemmatized tokens
for review in ner_reviews:
    review.pop("pos_tags", None)
    review.pop("ner_tags", None)
  
# step 10 remove non aspect words
# Remove words that are not aspects based on your domain-specific list or criteria
non_aspect_words = ["generic_word1", "generic_word2"]
for review in ner_reviews:
    review["filtered_tokens"] = [token for token in review["filtered_tokens"] if token not in non_aspect_words]

#step 11 handling negations
# Consider handling negations to capture changes in sentiment
for review in ner_reviews:
    for i, token in enumerate(review["lemmatized_tokens"]):
        if token == "not" and i < len(review["lemmatized_tokens"]) - 1:
            review["lemmatized_tokens"][i + 1] = "not_" + review["lemmatized_tokens"][i + 1]

# step 12 handling contractions
# Expand contractions for better tokenization
contractions = {"don't": "do not", "can't": "cannot", "won't": "will not"}
for review in ner_reviews:
    review["lemmatized_tokens"] = [contractions.get(token, token) for token in review["lemmatized_tokens"]]

# step 13(optipnal) contextual embeddings
# If using deep learning models, you may want to create contextual embeddings using pre-trained models (e.g., BERT)
# This typically involves using a specialized library like transformers or Hugging Face's transformers
# Example:
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
for review in ner_reviews:
    review["contextual_embeddings"] = tokenizer.encode(review["cleaned_text"], add_special_tokens=True)

# step 14 data splitting
# If not done earlier, split the data into training, validation, and test sets
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(ner_reviews, test_size=0.2, random_state=42)

### FEATURE EXTRACTION
#BOW (243-267)
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Sample dataset
reviews = [
    {"text": "The camera is great, but the battery life is disappointing.", "aspect_sentiment": {"camera": "positive", "battery": "negative"}, "preprocessed_tokens": ['camera', 'great', 'battery', 'life', 'disappointing']},
    {"text": "Easy to use and affordable.", "aspect_sentiment": {"usability": "positive", "price": "positive"}, "preprocessed_tokens": ['easy', 'use', 'affordable']},
    # Add more reviews to the dataset
]

# Extract preprocessed tokens from each review
preprocessed_reviews = [' '.join(review['preprocessed_tokens']) for review in reviews]

# Create a CountVectorizer instance
vectorizer = CountVectorizer()

# Fit and transform the preprocessed tokens
X_bow = vectorizer.fit_transform(preprocessed_reviews)

# Convert the sparse matrix to a DataFrame for better visualization
bow_df = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out())

# Display the BoW DataFrame
print(bow_df)

#TF-IDF (270-293)
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample dataset
reviews = [
    {"text": "The camera is great, but the battery life is disappointing.", "aspect_sentiment": {"camera": "positive", "battery": "negative"}, "preprocessed_tokens": ['camera', 'great', 'battery', 'life', 'disappointing']},
    {"text": "Easy to use and affordable.", "aspect_sentiment": {"usability": "positive", "price": "positive"}, "preprocessed_tokens": ['easy', 'use', 'affordable']},
    # Add more reviews to the dataset
]

# Extract preprocessed tokens from each review
preprocessed_reviews = [' '.join(review['preprocessed_tokens']) for review in reviews]

# Create a TfidfVectorizer instance
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the preprocessed tokens
X_tfidf = tfidf_vectorizer.fit_transform(preprocessed_reviews)

# Convert the sparse matrix to a DataFrame for better visualization
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Display the TF-IDF DataFrame
print(tfidf_df)

########MODEL
#NAIVE BAYES
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Assuming you have a target variable 'y' representing the sentiment labels for each review
# Replace 'y' with your actual target variable from your dataset

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Create and train the Naive Bayes model
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = naive_bayes_model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, predictions))

#SVM
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Assuming you have a target variable 'y' representing the sentiment labels for each review
# Replace 'y' with your actual target variable from your dataset

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Create and train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = svm_model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, predictions))


###MAKE PREDICTIONS ON NEW DATA
####MODEL EVALUATION
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

####MODEL INTERPRETATION
# For SVM, you can inspect coefficients for feature importance
coefficients = svm_model.coef_.toarray()
feature_names = tfidf_vectorizer.get_feature_names_out()

# Display top features
top_features = sorted(zip(coefficients[0], feature_names), reverse=True)[:10]
print("Top Features:")
for coef, feature in top_features:
    print(f"{feature}: {coef}")




