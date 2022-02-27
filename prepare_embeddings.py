import numpy as np

def document_vectorizer(corpus, model, num_features):
    """
    This averages all the word embeddings in the tweet. 
    This function averages all the word embeddings in the tweet.
    
    corpus: String text corpus
    Model: Model to use
    num_features: Int, the number of features to use
    
    """
    vocabulary = set(model.wv.index_to_key)
    
    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.
        
        for word in words:
            if word in vocabulary:
                nwords += 1
                feature_vector = np.add(feature_vector, model.wv[word])
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)
            
        return feature_vector
    
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                for tokenized_sentence in corpus]
    
    return np.array(features)


def document_vectorizer_glove(corpus, model, num_features):
    """
    This function averages all the word embeddings based on the glove model.

    corpus: String text corpus
    Model: Model to use
    num_features: Int, the number of features to use

    returns: numpy array


    """
    vocabulary = set(model.index_to_key)
    
    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.
        
        for word in words:
            if word in vocabulary:
                nwords += 1
                feature_vector = np.add(feature_vector, model[word])
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)
            
        return feature_vector
    
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                for tokenized_sentence in corpus]
    
    return np.array(features)