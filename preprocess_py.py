"""
Text preprocessing functions for sentiment analysis
"""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


class TextPreprocessor:
    """
    A comprehensive text preprocessing class for sentiment analysis
    """
    
    def __init__(self, use_stemming=False, use_lemmatization=True, remove_stopwords=True):
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        
        # Initialize NLTK tools
        if use_stemming:
            self.stemmer = PorterStemmer()
        if use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        
        # Get English stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Add common contractions
        self.contractions = {
            "ain't": "is not", "aren't": "are not", "can't": "cannot",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would", "he'll": "he will",
            "he's": "he is", "i'd": "i would", "i'll": "i will",
            "i'm": "i am", "i've": "i have", "isn't": "is not",
            "it'd": "it would", "it'll": "it will", "it's": "it is",
            "let's": "let us", "mustn't": "must not", "shan't": "shall not",
            "she'd": "she would", "she'll": "she will", "she's": "she is",
            "shouldn't": "should not", "that's": "that is", "there's": "there is",
            "they'd": "they would", "they'll": "they will", "they're": "they are",
            "they've": "they have", "we'd": "we would", "we're": "we are",
            "we've": "we have", "weren't": "were not", "what's": "what is",
            "where's": "where is", "who's": "who is", "won't": "will not",
            "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
            "you're": "you are", "you've": "you have"
        }
    
    def expand_contractions(self, text):
        """Expand contractions in text"""
        for contraction, expansion in self.contractions.items():
            text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
        return text
    
    def remove_html_tags(self, text):
        """Remove HTML tags from text"""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    def remove_urls(self, text):
        """Remove URLs from text"""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    
    def remove_special_characters(self, text, keep_punctuation=False):
        """Remove special characters and digits"""
        if keep_punctuation:
            # Keep basic punctuation
            pattern = r'[^a-zA-Z\s.!?]'
        else:
            pattern = r'[^a-zA-Z\s]'
        return re.sub(pattern, '', text)
    
    def normalize_whitespace(self, text):
        """Normalize whitespace"""
        return ' '.join(text.split())
    
    def preprocess_text(self, text, keep_punctuation=False):
        """
        Complete text preprocessing pipeline
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Expand contractions
        text = self.expand_contractions(text)
        
        # Remove HTML tags
        text = self.remove_html_tags(text)
        
        # Remove URLs
        text = self.remove_urls(text)
        
        # Remove special characters
        text = self.remove_special_characters(text, keep_punctuation)
        
        # Normalize whitespace
        text = self.normalize_whitespace(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming or lemmatization
        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        elif self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column='review', target_column='sentiment'):
        """
        Preprocess entire dataframe
        """
        df_processed = df.copy()
        
        # Preprocess text
        df_processed['processed_text'] = df_processed[text_column].apply(
            lambda x: self.preprocess_text(x)
        )
        
        # Add text features
        df_processed['text_length'] = df_processed[text_column].str.len()
        df_processed['word_count'] = df_processed[text_column].str.split().str.len()
        df_processed['processed_word_count'] = df_processed['processed_text'].str.split().str.len()
        
        # Encode target variable
        if target_column in df_processed.columns:
            df_processed['sentiment_encoded'] = df_processed[target_column].map({
                'positive': 1, 'negative': 0
            })
        
        return df_processed


def load_and_preprocess_data(data_path, test_size=0.2, random_state=42):
    """
    Load IMDB dataset and preprocess it for training
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        use_stemming=False,
        use_lemmatization=True,
        remove_stopwords=True
    )
    
    # Preprocess data
    df_processed = preprocessor.preprocess_dataframe(df)
    
    # Split data
    X = df_processed['processed_text']
    y = df_processed['sentiment_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, df_processed


def create_feature_vectors(X_train, X_test, method='tfidf', max_features=10000, ngram_range=(1, 2)):
    """
    Create feature vectors using TF-IDF or Count Vectorizer
    """
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=5,
            max_df=0.95
        )
    else:
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=5,
            max_df=0.95
        )
    
    # Fit and transform training data
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Transform test data
    X_test_vectorized = vectorizer.transform(X_test)
    
    return X_train_vectorized, X_test_vectorized, vectorizer


def get_sentiment_features(text):
    """
    Extract sentiment-related features using TextBlob
    """
    blob = TextBlob(text)
    
    features = {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'word_count': len(text.split()),
        'char_count': len(text),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    }
    
    return features


def add_advanced_features(df, text_column='processed_text'):
    """
    Add advanced features to the dataframe
    """
    df_features = df.copy()
    
    # Apply sentiment features
    sentiment_features = df_features[text_column].apply(get_sentiment_features)
    
    # Convert to dataframe and add to original
    features_df = pd.DataFrame(sentiment_features.tolist())
    
    for col in features_df.columns:
        df_features[f'feature_{col}'] = features_df[col]
    
    return df_features


if __name__ == "__main__":
    # Example usage
    data_path = "/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, df_processed = load_and_preprocess_data(data_path)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Sample processed text: {X_train.iloc[0]}")
    
    # Create feature vectors
    X_train_vec, X_test_vec, vectorizer = create_feature_vectors(X_train, X_test)
    
    print(f"Feature matrix shape: {X_train_vec.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
