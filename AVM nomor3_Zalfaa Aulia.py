# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 08:44:54 2022

@author: Lenovo X380 Yoga
"""

# Define the documents
sent1 = "I love sky , I love sea"
sent2 = "I like running , I love reading"

sentences = [sent1, sent2]

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Create the Document Term Matrix
count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer = CountVectorizer()
sparse_matrix = count_vectorizer.fit_transform(sentences)

# OPTIONAL: Convert Sparse Matrix to Pandas Dataframe if you want to see the word frequencies.
doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(doc_term_matrix, 
                  columns=count_vectorizer.get_feature_names(), 
                  index=['sent1', 'sent2', 'sent3'])
df
# Compute Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(df, df))