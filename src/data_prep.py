def get_wordnet_pos(treebank_tag):
    """Map POS tag to first character lemmatize() accepts."""
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

def clean_text(review, stop_words=None, lemmatize=True):
    """Clean and preprocess a single review text."""
    tokenizer = RegexpTokenizer(r"([a-zA-Z]+(?:’[a-z]+)?)")
    lemmatizer = WordNetLemmatizer() if lemmatize else None
    
    tokens = tokenizer.tokenize(review.lower())
    if lemmatize:
        pos_tags = pos_tag(tokens)
        tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    if stop_words:
        tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def preprocess_texts(reviews, stop_words=None, lemmatize=False):
    """Apply text cleaning and preprocessing to a list of texts."""
    return [clean_text(review, stop_words=stop_words, lemmatize=lemmatize) for review in reviews]