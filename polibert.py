import nltk
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer, pipeline

# Choose GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select mode path here
pretrained_LM_path = "kornosk/polibertweet-mlm"

# Load model
tokenizer = AutoTokenizer.from_pretrained(pretrained_LM_path)
model = AutoModel.from_pretrained(pretrained_LM_path)

# upload csv file
annotated_data = pd.read_csv('./ExtractedTweets.csv')
features = annotated_data['Tweet']
labels = annotated_data['Party']
# print(features)

# Tokenize and generate embeddings for each tweet
tweet_embeddings = []
for tweet in features:
    # Tokenize the tweet
    encoded_input = tokenizer(tweet, return_tensors="pt", padding=True, truncation=True)
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**encoded_input)
    # Extract the embeddings of the first token ([CLS]) as the tweet-level embedding
    tweet_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    tweet_embeddings.append(tweet_embedding)

tweet_embeddings = torch.stack(tweet_embeddings)

# Now `tweet_embeddings` contains the embeddings of each tweet
print(tweet_embeddings[0])


# # Load pre-trained PoliBERT model and tokenizer
# model_name = "nlpaueb/polibert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)

# annotated_data = pd.read_csv('./ExtractedTweets.csv')

# # Tokenize and encode input
# inputs = tokenizer(annotated_data, return_tensors="pt", padding=True, truncation=True)

# # Pass input through PoliBERT
# with torch.no_grad():
#     outputs = model(**inputs)

# # Get word embeddings
# word_embeddings = outputs.last_hidden_state

# # Split the dataset into train and test sets
# train_data, test_data = train_test_split(annotated_data, test_size=0.8, random_state=42)

# # Fine-tune and train PoliBERT model on the training set (Not included, assume it's done)

# # Use the trained model to make predictions on the test set
# # For simplicity, let's assume the model.predict() method is available
# predicted_labels = model.predict(test_data)

# # Extract ground truth labels from the test set (assuming your data is annotated with labels)
# # For simplicity, let's assume you have a function to extract labels from your dataset
# true_labels = extract_true_labels(test_data)

# # Calculate accuracy
# accuracy = accuracy_score(true_labels, predicted_labels)

# print("Accuracy:", accuracy)