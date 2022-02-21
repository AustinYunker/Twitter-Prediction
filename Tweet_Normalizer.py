import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import requests
import nltk
import unicodedata
from nltk.tokenize.toktok import ToktokTokenizer

def strip_html_tags(text):
    """
    This function removes unnecessary HTML tags in the corpus. 
    
    test: corpus of text data
    
    returns: text with no HTML

    """
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(["iframe", "script"])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    
    return stripped_text

def text_tokenizer(text):
    """
    This function takes a corpus and tokenizes it into sentences
    Parameters
    
    test: String containing the corpus
    
    returns: Corpus of tokenized sentences

    """
    tokenizer = nltk.sent_tokenize
    sentence_tokens = tokenizer(text=text)
    return sentence_tokens 


def word_tokenizer(text):
    """
    This function takes a corpus and splits the sentences into words
    
    test: String of tokenized sentences
    
    returns: Array of tokenized words

    """
    tokenizer = nltk.word_tokenize
    word_tokens = tokenizer(text)
    return np.array(word_tokens)

def remove_accented_chars(text):
    """
    This function removes accented characters from the corpus using ASCII
    characters
    
    text: String corpus of text data
    
    returns: Corpus with all characters converted and standardized into ASCII characters

    """
    
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

#This map was gotten from the internet

CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"}

def expand_contractions(text, contraction_mapping = CONTRACTION_MAP):
    """
    This function takes a corpus of text data and expands all contractions
    bassed on the contraction mapping
    
    text: String corpus of text data
    
    returns: Text corpus with all contractions expanded

    """
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                    else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction
    
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def remove_special_characters(text, remove_digits = False):
    """
    This function returns the text corpus with all special characters removed with the option
    to remove digits
    
    text: String corpus
    
    returns: Text corpus with special characters removed

    """
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

def case_conversation(text, text_lower = True):
    """
    This function returns the corpus with all tokens lowercase or uppercase
    
    text: String corpus
    text_lower: Boolen, default is True
    
    returns: Text corpus with all tokens converted

    """
    if text_lower:
        return text.lower()
    else:
        return text.upper()
    
nlp = spacy.load('en_core_web_sm')
def lemmatize(text):
    """
    This function returns the lemmatized text corpus. Gives correct spelling. 

    text: String corpus
    
    returns: String of lemmatized text corpus
    """
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def remove_stopwords(text, is_lower_case=False):
    """
    This function removes all the stopwords from the text corpus based on the pre-existing 
    list of stopwords from nltk. 
    
    text: String corpus
    is_lower_case: Boolean to make the corpus lower case
    
    returns: String corpus with stopwords removed

    """
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        
    filtered_text = ' '.join(filtered_tokens)
    
    return filtered_text

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')

def normalize_corpus(corpus, html_stripping = True, contraction_expansion = True, accented_char_removal = True, 
                     text_lower_case = True, text_lemmatization = True, special_char_removal = True, stopword_removal=True, 
                     remove_digits = True):
    """
    This function normalizes the text corpus based on all the functions above. 

    corpus : Text corpus
    html_stripping : boolean, optional
        Controls if the text corpus is stripped of htmls
    contraction_expansion : boolean, optional
        Controls if the text corpus has its contractions extraction
    accented_char_removal : boolean, optional
        Controls if the text corpus has its accented characters removed
    text_lower_case : boolean, optional
        Controls if the text corpus has its text changed to lower case
    correct_spelling : boolean, optional
        Controls if the text corpus has its spelling corrected
    text_lemmatization : boolean, optional
        Controls if the text corpus his its words lemmatized
    special_char_removal : boolean, optional
        Controls if the text corpus has its special characters removed
    stopword_removal : boolean, optional
        Controls if the text corpus has its stopwords removed
    remove_digits : boolean, optional
        Controls if the text corpus has its digits removed

    returns: String of normalized text corpus
    """
    
    normalized_corpus = []

    for doc in corpus:
        
        if html_stripping:
            doc = strip_html_tags(doc)
            
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        if contraction_expansion:
            doc = expand_contractions(doc)
            
        if text_lower_case:
            doc = case_conversation(doc)
            
        #Remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)
            
        if text_lemmatization:
            doc = lemmatize(doc)
            
        if special_char_removal:
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)
            
        #Removes extra whitespace
        doc = re.sub(' +', ' ', doc)
        
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
