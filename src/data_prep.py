from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin

class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words=None, lemmatize=True):
        """Initialize the text cleaner transformer."""
        self.stop_words = stop_words
        self.lemmatize = lemmatize
        self.lemmatizer = None
        self.tokenizer = RegexpTokenizer(r"([a-zA-Z]+(?:’[a-z]+)?)")

    def fit(self, X, y=None):
        """Fit method. This transformer does not require fitting."""
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        if self.stop_words is None:
            self.stop_words = set()
        else:
            self.stop_words = set(self.stop_words)
        return self

    def transform(self, X, y=None):
        """Transform the input data by cleaning the text."""
        return [self.clean_text(review) for review in X]

    def clean_text(self, review):
        """Clean and preprocess a single review text."""
        tokens = self.tokenizer.tokenize(review.lower())
        if self.lemmatize and self.lemmatizer:
            pos_tags = pos_tag(tokens)
            tokens = [self.lemmatizer.lemmatize(word, self.get_wordnet_pos(tag))
                      for word, tag in pos_tags]
        tokens = [word for word in tokens if word not in self.stop_words]

        return ' '.join(tokens)

    def get_wordnet_pos(self, treebank_tag):
        """Map POS tag to the format required by the lemmatizer."""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        return wordnet.NOUN

import numpy as np
from tensorflow.keras.layers import TextVectorization
from sklearn.base import BaseEstimator, TransformerMixin

class KerasTextVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Initialize the TextVectorization layer.
        
        :param max_tokens: Maximum size of the vocabulary.
        :param output_sequence_length: Length of the output sequences.
        """
        self.text_vectorization = TextVectorization(
            standardize=None, 
            max_tokens=20000,
            output_mode='int',
            output_sequence_length=200)

    def adapt(self, texts):
        """
        Fit the TextVectorization layer to the texts.
        """
        self.text_vectorization.adapt(texts)

    def fit(self, X, y=None):
        """
        Fit the TextVectorization layer to the training data.
        
        :param X: Iterable over raw text data.
        :param y: Not used, present for API consistency by convention.
        """
        self.adapt(X)
        return self

    def transform(self, X, y=None):
        """
        Transform the text data to vectors.
        
        :param X: Iterable over raw text data.
        :param y: Not used, present for API consistency by convention.
        """
        return self.text_vectorization(X).numpy()
