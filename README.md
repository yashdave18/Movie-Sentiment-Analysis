# Movie Sentiment Analysis

## Overview

This project performs binary sentiment classification on IMDB movie reviews, predicting whether a given review is positive or negative. It uses classical NLP preprocessing techniques combined with Naive Bayes classifiers.

## Dataset

The IMDB Dataset contains 50,000 movie reviews labeled as positive or negative. Sentiment labels are encoded numerically — positive as 1 and negative as 0 — before any processing begins.

## Text Preprocessing

Raw movie reviews contain noise that must be cleaned before any model can learn from them. The preprocessing pipeline applies the following steps in order.

HTML tags are stripped first since the reviews were scraped from the web and contain residual markup. All text is then converted to lowercase to ensure words like "Good" and "good" are treated identically. Special characters and punctuation are replaced with spaces so only alphanumeric tokens remain. Common English stopwords such as "the", "is", and "and" are removed using NLTK since they carry no sentiment signal. Finally, Porter Stemming reduces each word to its root form — for example, "running" becomes "run" — to consolidate variations of the same word.

## Feature Extraction

The cleaned reviews are converted into numerical vectors using a Bag of Words approach via scikit-learn's CountVectorizer, retaining the top 1000 most frequent tokens. Each review is represented as a vector of word counts across this vocabulary.

## Models

Three variants of the Naive Bayes classifier are trained and compared. GaussianNB assumes features follow a Gaussian distribution, which is not well-suited for count-based text data. MultinomialNB is designed for discrete count features and is the standard choice for Bag of Words representations. BernoulliNB treats features as binary indicators of word presence or absence.

## Results

GaussianNB achieved an accuracy of 0.80 with a macro F1-score of 0.80, performing the weakest of the three due to its poor fit with count-based features. MultinomialNB and BernoulliNB both achieved an accuracy of 0.84 with a macro F1-score of 0.84. BernoulliNB showed a slightly higher recall of 0.87 on the positive class compared to 0.85 for MultinomialNB.

The confusion matrix for MultinomialNB shows 4166 true negatives, 4227 true positives, 847 false positives, and 760 false negatives out of 10,000 test samples, indicating balanced performance across both classes.

