from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords  
import nltk
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


#Data cleaning
# def get_wordnet_pos(treebank_tag):
#     """Map POS tag to first character lemmatize() accepts."""
#     if treebank_tag.startswith('J'):
#         return wordnet.ADJ
#     elif treebank_tag.startswith('V'):
#         return wordnet.VERB
#     elif treebank_tag.startswith('N'):
#         return wordnet.NOUN
#     elif treebank_tag.startswith('R'):
#         return wordnet.ADV
#     else:
#         return wordnet.NOUN

# def clean_text(review, stop_words=None, lemmatize=True):
#     """Clean and preprocess a single review text."""
#     tokenizer = RegexpTokenizer(r"([a-zA-Z]+(?:’[a-z]+)?)")
#     lemmatizer = WordNetLemmatizer() if lemmatize else None
    
#     tokens = tokenizer.tokenize(review.lower())
#     if stop_words:
#         tokens = [word for word in tokens if word not in stop_words]
#     if lemmatize:
#         pos_tags = pos_tag(tokens)
#         tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
#     return ' '.join(tokens)


# def preprocess_texts(reviews, stop_words=None, lemmatize=False):
#     """Apply text cleaning and preprocessing to a list of texts."""
#     return [clean_text(review, stop_words=stop_words, lemmatize=lemmatize) for review in reviews]