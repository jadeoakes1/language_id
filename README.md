# language_id
This Python script classifies written text into different languages using Naive Bayes, Logistic Regression, and Random Forest classifiers. It preprocesses the dataset, trains models on various text features, and evaluates their performance on training, development, and test sets.

## Language Identification Final Project - README
**Author:** Jade Oakes  
**Date:** December 12, 2023

## Overview
This Python program is a final project for language identification using machine learning algorithms such as Naive Bayes, Logistic Regression, and Random Forest. It aims to classify written text into different languages based on features extracted from the text data.

## Requirements
- Python 3.x
- Pandas
- NumPy
- NLTK
- Scikit-learn

## Usage
Ensure that all required libraries are installed. You can install them using pip:
```bash
pip install pandas numpy nltk scikit-learn
```

Download the NLTK data for tokenization:
```bash
import nltk
nltk.download('punkt')
```

Prepare your dataset in CSV format. The dataset should contain a column with text data and a column indicating the language of each text sample.

Update the dataset.csv filename in the script to match your dataset.

Run the script:
```bash
python language_id.py
```


## Functionality
The program splits the dataset into training, development (dev), and test sets.
It tokenizes text data into unigrams, characters, and character bigrams.
Features are vectorized using DictVectorizer.
Three machine learning algorithms are used: Naive Bayes, Logistic Regression, and Random Forest.
The program performs hyperparameter tuning and evaluates the models on the dev set.
Finally, it evaluates the best-performing models on the test set and prints classification reports and accuracy scores.

## Output
The program prints classification reports and accuracy scores for each model and feature type. Results are displayed for default hyperparameters as well as modified hyperparameters. Output for Naive Bayes, Logistic Regression, and Random Forest models are separated for easy interpretation.

## Additional Notes
Make sure to customize hyperparameters, such as alpha for Naive Bayes and C for Logistic Regression, based on your dataset and performance requirements.
Experiment with different feature types (unigrams, characters, character bigrams) to see which yields the best results for your dataset.

## Project Results
Processed 11,000 text samples in 11 languages.
Achieved 97.8% accuracy using Logistic Regression with unigram features.
Detailed results and methodology can be found in the [full writeup](language_id_writeup.pdf).

## Repository Contents
- `language_id.py`: Source code for the project.
- `dataset.csv`: Sample data used for training and testing.
- `language_id_writeup.pdf`: Full project writeup.

## Contact
For any questions or collaborations, please reach out to oakesjade@gmail.com.

##

This README provides an overview of the Language Identification Final Project script. For further details on the code implementation and functionality, please refer to the comments within the script itself.
