import numpy as np
import pandas as pd
from joblib import dump
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import RegexpTokenizer
import nltk
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from tensorflow.data import AUTOTUNE as tf_AUTOTUNE, Dataset as tf_Dataset
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, GRU
from tensorflow.keras.models import Sequential
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

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

def clean_text(review, tokenizer, stop_words=None, lemmatize=False, tokenize=False):
    """Clean and preprocess a single review text."""
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

def preprocess_texts(reviews, tokenizer, stop_words=None, lemmatize=False, tokenize=False):
    """Apply optimized text cleaning and preprocessing to a list of texts."""
    return [clean_text(review, tokenizer, stop_words=stop_words, lemmatize=lemmatize, tokenize=tokenize) for review in reviews]

def prepare_tf_dataset(X, y, batch_size, is_training=False):
    """
    Prepares a TensorFlow dataset for efficient training or evaluation.
    """
    dataset = tf_Dataset.from_tensor_slices((X, y))
    if is_training:
        dataset = dataset.shuffle(10000)  # Shuffle only if dataset is for training
    return dataset.batch(batch_size).cache().prefetch(tf_AUTOTUNE)

def extract_performance_metrics(history, callbacks):
    early_stopping = next(
        (cb for cb in callbacks if isinstance(cb, EarlyStopping)), 
        None
    )
    if early_stopping and early_stopping.stopped_epoch > 0:
        adjusted_epoch = early_stopping.stopped_epoch - early_stopping.patience
        max_epoch_index = len(history.history['loss']) - 1
        best_epoch = max(0, min(adjusted_epoch, max_epoch_index))
    else:
        best_epoch = len(history.history['loss']) - 1

    metrics = {
        'loss': history.history['loss'][best_epoch],
        'val_loss': history.history['val_loss'][best_epoch],
        'val_accuracy': history.history.get('val_accuracy', [None])[best_epoch],
        'val_auc': history.history.get('val_auc', [None])[best_epoch]
    }
    return metrics

def bag_of_words_CV(model_name, model, pipe, X_train, y_train, cv):
    model_pipeline = Pipeline(
        steps=pipe.steps + [
            (model_name, model)
        ]
    )
    cv_results = cross_validate(model_pipeline, 
                            X_train, 
                            y_train, 
                            cv=cv, 
                            scoring=['accuracy', 'roc_auc'], 
                            return_train_score=False)
    
    model_pipeline.fit(X_train, y_train)
    
    dump(model_pipeline, model_name + '_pipeline.joblib')
    
    accuracy = cv_results['test_accuracy'].mean()
    auc = cv_results['test_roc_auc'].mean()
    df = pd.DataFrame([[accuracy,auc]],columns=['Accuracy','AUC'], index=[model_name])
    return df

def keras_cv(model_name,X,y,cv,tokenizer,text_vectorization,CALLBACKS, glove=False, glove_path = None):
    metrics_aggregate = {'loss': 0, 'val_loss': 0, 'val_accuracy': 0, 'val_auc': 0}
    X = np.array(X)
    y = np.array(y)
    runs = 0
    for train, validation in cv.split(X, y):
        runs += 1
        X_train = X[train]
        y_train = y[train]
        X_val = X[validation]
        y_val = y[validation]
    
        X_train_clean = preprocess_texts(X_train, tokenizer)
        X_val_clean = preprocess_texts(X_val, tokenizer)
    
        text_vectorization.adapt(X_train)
        X_train = text_vectorization(X_train)
        X_val = text_vectorization(X_val)
    
        train_ds = prepare_tf_dataset(X_train, y_train, 256, is_training=True)
        val_ds = prepare_tf_dataset(X_val, y_val, 256)

        if glove:
            vocabulary = text_vectorization.get_vocabulary()
            vocab_size = len(vocabulary)
            
            # Load GloVe embeddings from file.
            glove_embeddings = {}
            with open(glove_path, 'r', encoding='utf-8') as file:
                for line in file:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], dtype='float32')
                    glove_embeddings[word] = vector

            # Initialize the embedding matrix with zeros.
            embedding_matrix = np.zeros((vocab_size, 300))
            
            # Populate the embedding matrix with GloVe vectors.
            for i, word in enumerate(vocabulary):
                embedding_vector = glove_embeddings.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
                    
            model = Sequential([Embedding(input_dim=20000, output_dim=300, input_length=200, weights=[embedding_matrix], trainable=False),
                                Bidirectional(GRU(32)),
                                Dropout(0.4),
                                Dense(16, activation='relu'),
                                Dropout(0.4),
                                Dense(1, activation='sigmoid')
                                ])
        
        model = Sequential([Embedding(input_dim=20000, output_dim=32, input_length=200),
                            Bidirectional(GRU(16)),
                            Dense(8, activation='relu'),
                            Dense(1, activation='sigmoid')
    ])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'AUC'])
        
        results = model.fit(train_ds,
                            validation_data= val_ds,
                            epochs=100,
                            verbose=0,
                            callbacks=CALLBACKS)
        
        metrics = extract_performance_metrics(results, CALLBACKS)
        for key in metrics_aggregate:
                metrics_aggregate[key] += metrics[key]
    results_dic = {key: val / runs for key, val in metrics_aggregate.items()}
    accuracy = results_dic['val_accuracy']
    auc = results_dic['val_auc']
    df = pd.DataFrame([[accuracy,auc]],columns=['Accuracy','AUC'], index=[model_name])
    return pd.DataFrame([results_dic])