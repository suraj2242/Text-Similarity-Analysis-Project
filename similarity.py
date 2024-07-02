import os
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to preprocess text
def preprocess_text(text):
    # Convert the text to lower case
    text = text.lower()
    
    # Remove punctuation from the text
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

# List of text file names
file_names = ['dataMiningthings/file1.txt', 'dataMiningthings/file2.txt', 'dataMiningthings/file3.txt', 'dataMiningthings/file4.txt', 'dataMiningthings/file5.txt']

# List to store the text in each file
texts = []

# Read each text file
for file_name in file_names:
    with open(file_name, 'r') as file:
        text = file.read()
        text = preprocess_text(text)
        texts.append(text)
        

# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Vectorize the text
X = vectorizer.fit_transform(texts)
# print(X)
# Convert the sparse matrix to a dense matrix
X_dense = X.toarray()


# print(X_dense)
# Input text
input_text = input("Enter your text: ")

# Preprocess the input text
input_text = preprocess_text(input_text)


# Vectorize the input text
input_vector = vectorizer.transform([input_text])
input_dense = input_vector.toarray()

# Calculate the cosine similarity between the input text and the documents
similarity_scores = cosine_similarity(input_dense, X_dense).flatten()

# Get the indices of the documents sorted by similarity
#argsort returns the indices that would sort the similarity_scores array in ascending order
sorted_indices = similarity_scores.argsort()[::-1]

# Print the top 3 documents
for index in sorted_indices[:3]:
    print(f"Document: {file_names[index]}, Similarity: {similarity_scores[index]}")