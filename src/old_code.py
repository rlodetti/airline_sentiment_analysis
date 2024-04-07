import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
# from scikeras.wrappers import KerasClassifier
from tensorflow.data import Dataset as tf_dataset, AUTOTUNE
from tensorflow.keras.models import Sequential
from tensorflow.keras. layers import TextVectorization
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

import nltk
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

def get_wordnet_pos_optimized(treebank_tag):
    """Map POS tag to first character lemmatize() accepts."""
    tag_dict = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }
    # Default to NOUN if not found
    return tag_dict.get(treebank_tag[0], wordnet.NOUN)

def clean_text(review, stop_words=None, lemmatize=False, tokenize=False):
    """Clean and preprocess a single review text."""
    # Move these initializations outside if called multiple times
    tokenizer = RegexpTokenizer(r"([a-zA-Z]+(?:’[a-z]+)?)")
    tokens = tokenizer.tokenize(review.lower())
    
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        pos_tags = pos_tag(tokens)
        tokens = [lemmatizer.lemmatize(word, get_wordnet_pos_optimized(tag)) for word, tag in pos_tags]
    
    if stop_words:
        stop_words_set = set(stop_words)
        tokens = [word for word in tokens if word not in stop_words_set]
    
    if tokenize:
        return tokens
    else:
        return ' '.join(tokens)

def preprocess_texts(reviews, stop_words=None, lemmatize=False, tokenize=False):
    """Apply optimized text cleaning and preprocessing to a list of texts."""
    return [clean_text(review, stop_words=stop_words, lemmatize=lemmatize, tokenize=tokenize) for review in reviews]


class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words=None, lemmatize=True):
        self.stop_words = set(stop_words) if stop_words else None
        self.lemmatize = lemmatize
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.tokenizer = RegexpTokenizer(r"([a-zA-Z]+(?:’[a-z]+)?)")
        
    def fit(self, X, y=None):
        return self  # No fitting necessary for this transformer
    
    def transform(self, X, y=None):
        return [self.clean_text(review) for review in X]
    
    def clean_text(self, review):
        tokens = self.tokenizer.tokenize(review.lower())  # Tokenize and lowercase
        if self.lemmatize:
            pos_tags = pos_tag(tokens)
            tokens = [self.lemmatizer.lemmatize(word, self.get_wordnet_pos(tag))
                      for word, tag in pos_tags]
        if self.stop_words:
            tokens = [word for word in tokens if word not in self.stop_words]
        
        return ' '.join(tokens)
    
    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words=None, lemmatize=True):
        self.stop_words = stop_words
        self.lemmatize = lemmatize

    def fit(self, X, y=None):
        # Initialize tokenizer and lemmatizer here if necessary, but it seems you don't need to fit anything
        return self  # No fitting necessary for this transformer

    def transform(self, X, y=None):
        # Lazy initialization of components
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = RegexpTokenizer(r"([a-zA-Z]+(?:’[a-z]+)?)")
        if self.lemmatize and not hasattr(self, 'lemmatizer'):
            self.lemmatizer = WordNetLemmatizer()
        
        # Ensure stop_words is a set for faster lookup
        stop_words_set = set(self.stop_words) if self.stop_words else None
        
        return [self.clean_text(review, stop_words_set) for review in X]
    
    def clean_text(self, review, stop_words_set):
        tokens = self.tokenizer.tokenize(review.lower())  # Tokenize and lowercase
        if self.lemmatize:
            pos_tags = pos_tag(tokens)
            tokens = [self.lemmatizer.lemmatize(word, self.get_wordnet_pos(tag))
                      for word, tag in pos_tags]
        if stop_words_set:
            tokens = [word for word in tokens if word not in stop_words_set]
        
        return ' '.join(tokens)
    
    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

def summarize_mlp_grid_search_results(grid_search):
    columns_to_extract = [
        ('mean_fit_time', 'fit_time'),
        ('mean_score_time', 'score_time'),
        ('param_mlp__model__num_layers', 'num_layers'),
        ('param_mlp__model__units', 'units'),
        ('mean_test_score', 'balanced_accuracy'),
        ('param_mlp__model__initializer','initializer'),
        ('param_mlp__model__dropout_rate', 'dropout_rate')
        
    ]
    summary_df = pd.DataFrame(grid_search.cv_results_)[[original for original, renamed in columns_to_extract]]

    summary_df.columns = [renamed for original, renamed in columns_to_extract]
    
    # Calculate total time and convert to int
    summary_df['time'] = (summary_df['fit_time'] + summary_df['score_time']).astype(int)
    
    # Reorder and select final columns for the output
    final_columns = ['balanced_accuracy', 'time', 'num_layers', 'units', 'dropout_rate', 'initializer']
    final_df = summary_df[final_columns]
    sorted_df = final_df.sort_values(by=['balanced_accuracy', 'time'], ascending=[False, True])
    
    return sorted_df

def summarize_rnn_grid_search_results(grid_search):
    columns_to_extract = [
        ('mean_fit_time', 'fit_time'),
        ('mean_score_time', 'score_time'),
        ('param_rnn__model__bi_directional', 'bi_directional'),
        ('param_rnn__model__dense_layers', 'num_dense_layers'),
        ('param_rnn__model__recurrent_type', 'recurrent_type'),
        ('param_rnn__model__rnn_layers', 'num_rnn_layers'),
        ('param_rnn__model__units', 'units'),
        ('param_rnn__model__dropout_rate', 'dropout_rate'),
        ('mean_train_score', 'train_score'),
        ('mean_test_score', 'test_score')
    ]
    summary_df = pd.DataFrame(grid_search.cv_results_)[[original for original, renamed in columns_to_extract]]

    summary_df.columns = [renamed for original, renamed in columns_to_extract]
    
    # Calculate total time and convert to int
    summary_df['time'] = (summary_df['fit_time'] + summary_df['score_time']).astype(int)
    
    # Reorder and select final columns for the output
    final_columns = ['train_score', 'test_score', 'time', 'units', 'bi_directional', 'recurrent_type', 'num_rnn_layers', 'num_dense_layers', 'dropout_rate']
    final_df = summary_df[final_columns]
    sorted_df = final_df.sort_values(by=['test_score', 'time'], ascending=[False, True])
    
    return sorted_df

def load_data(path, val=False, sample=0):
    if sample == 0:
        df = pd.read_csv(path)[['Review_Title','Review','Recommended']]
    else:
        df = pd.read_csv(path)[['Review_Title','Review','Recommended']].sample(sample)
    
    X = df['Review_Title'] + ' ' + df['Review']
    y = df['Recommended'].map({'yes':1,'no':0})
    if val:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)
        return X_train, X_test, y_train, y_test

def create_tf_datasets(X, y, is_training=False):
    """
    Converts selected features and labels into TensorFlow datasets.
    """
    if is_training:
        train_ds = tf_dataset.from_tensor_slices((np.array(X), y.astype('float32')))
        train_ds = train_ds.shuffle(1000, reshuffle_each_iteration=True).batch(64).cache().prefetch(AUTOTUNE)
        return train_ds
    else:
        test_ds = tf_dataset.from_tensor_slices((np.array(X), y.astype('float32')))
        test_ds = test_ds.batch(64).cache().prefetch(AUTOTUNE)
        return test_ds

class KerasTextVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_tokens, output_sequence_length):
        self.max_tokens = max_tokens
        self.output_sequence_length = output_sequence_length
        self.text_vectorization = TextVectorization(
            max_tokens=self.max_tokens,
            output_sequence_length=self.output_sequence_length)

    def fit(self, X, y=None):
        self.text_vectorization.adapt(X)
        return self  # Return self to allow chaining

    def transform(self, X, y=None):
        return self.text_vectorization(X).numpy()  # Convert to numpy for sklearn compatibility