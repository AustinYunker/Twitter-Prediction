# Twitter-Prediction

#### Project Status: [Completed as of February 27th, 2022]

## Project Intro/Objective
The purpose of this project is to classify tweets as either specifying a real natural disaster or a non-disaster (safe). 


### Methods Used
* Data Visualization
* Machine Learning
  * Logistic Regression
  * Naive Bayes
  * Random Forest
  * XGBoost
  * Deep Learning

### Technologies 
* Python (pandas, numpy, matplotlib, tensorflow, keras)
* Google BigQuery


## Project Description
The data used in this project came from a Kaggle competition and can be accessed here [here](https://www.kaggle.com/c/nlp-getting-started/overview). The performance metric used in this competition was the F1-score. To get hands on experience working with Text data, three models were choosen based on features generated from dense words embeddings. To generate these embeddings, Word2Vec, FastText, and GloVe were used. Each embedding was used as input to all models that were considered. Before generating the dense embeddings, the training tweets were normalized and mispelled/invalid words were dropped. The best model for all three embeddings was XGBoost. To get the test predictions, a vote was taken for the majority class. Thus, if Word2Vec and FastText models predicted a tweet was a disaster and the GloVe model predicted a safe tweet, the vote would be a disaster tweet. Finally, the test predictions were submitted to Kaggle to be scored. The test score was an F1-score of 79%. 


## Getting Started

1. Fork and then Clone this repo
2. Create an Access Key for Kaggle to download the data using the Kaggle API. Will need to have a Kaggle account to do this. 


## Featured Notebooks
* [Cleaning the Tweets](https://github.com/AustinYunker/Twitter-Prediction/blob/main/Tweet%20Cleaner.ipynb)
* [Visualizing the Tweets](https://github.com/AustinYunker/Twitter-Prediction/blob/main/Tweet%20Visualization.ipynb)
* [Submitting Test Predictions](https://github.com/AustinYunker/Twitter-Prediction/blob/main/Model%20Test%20Predictions.ipynb)



## Contact
* Feel free to contact me with any questions or if you are interested in contributing!

## Acknowledgements 
* This style of README was adopted from Rocio Ng and the original template can be found [here](https://github.com/sfbrigade/data-science-wg/blob/master/dswg_project_resources/Project-README-template.md)
