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


