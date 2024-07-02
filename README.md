# Text-Similarity-Analysis-Project
## This Python script reads text from multiple files, preprocesses it, and computes the cosine similarity between a user-input text and the texts in the files. It then prints the top 3 most similar documents.

## Requirements
1. Python 3.x
2. pandas
3. scikit-learn

## Setup
1. Ensure you have Python 3.x installed on your machine.
2. Install the required libraries using pip: ```pip install pandas scikit-learn```
3. Place the text files you want to compare in the dataMiningthings directory with filenames file1.txt, file2.txt, file3.txt, file4.txt, and file5.txt.

## Usage
1. Save the script as similarity.py.
2. Run the script using the command: ```python similarity.py```
3. Enter the text you want to compare when prompted.
4. The script will print the filenames of the top 3 most similar documents along with their similarity scores.

## Script Details
### Text Preprocessing
  1. Converts text to lower case.
  2. Removes punctuation.
     
### Vectorization
1. Uses CountVectorizer from scikit-learn to convert text to vectors.
   
### Cosine Similarity
1. Computes the cosine similarity between the input text vector and the vectors of the document texts.
2. Prints the top 3 documents with the highest similarity scores.
