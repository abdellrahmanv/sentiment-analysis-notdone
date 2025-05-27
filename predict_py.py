"""
Prediction functions for the Streamlit app
"""

import joblib
import pandas as pd
import numpy as np
from preprocess import TextPreprocessor
import os
from textblob import TextBlob
import re


class SentimentPredictor:
    """
    A class for making sentiment predictions using trained models
    """
    
    def __init__(self, model_path=None, vectorizer_path=None, model_info_path=None):
        self.model = None
        self.vectorizer = None
        self.model_info = None
        self.preprocessor = TextPreprocessor(
            use_stemming=False,
            use_lemmatization=True,
            remove_stopwords=True
        )
        
        # Load model components if paths are provided
        if model_path and vectorizer_path:
            self.load_model(model_path, vectorizer_path, model_info_path)
    
    def load_model(self, model_path, vectorizer_path, model_info_path=None):
        """Load the trained model and vectorizer"""
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            
            if model_info_path and os.path.exists(model_info_path):
                self.model_info = joblib.load(model_info_path)
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_single_text(self, text):
        """Preprocess a single text for prediction"""
        if not text or pd.isna(text):
            return ""
        
        # Use the preprocessor
        processed_text = self.preprocessor.preprocess_text(text)
        return processed_text
    
    def predict_sentiment(self, text):
        """Predict sentiment for a single text"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model and vectorizer must be loaded first")
        
        # Preprocess the text
        processed_text = self.preprocess_single_text(text)
        
        # Vectorize the text
        text_vectorized = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.model.predict(text_vectorized)[0]
        probability = self.model.predict_proba(text_vectorized)[0]
        
        # Convert prediction to sentiment label
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = max(probability)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_prob': probability[1],
            'negative_prob': probability[0],
            'processed_text': processed_text
        }
    
    def predict_batch(self, texts):
        """Predict sentiment for multiple texts"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model and vectorizer must be loaded first")
        
        # Preprocess all texts
        processed_texts = [self.preprocess_single_text(text) for text in texts]
        
        # Vectorize all texts
        texts_vectorized = self.vectorizer.transform(processed_texts)
        
        # Make predictions
        predictions = self.model.predict(texts_vectorized)
        probabilities = self.model.predict_proba(texts_vectorized)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            sentiment = "Positive" if pred == 1 else "Negative"
            confidence = max(prob)
            
            results.append({
                'text': texts[i],
                'sentiment': sentiment,
                'confidence': confidence,
                'positive_prob': prob[1],
                'negative_prob': prob[0],
                'processed_text': processed_texts[i]
            })
        
        return results
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model_info:
            return self.model_info
        else:
            return {
                'model_name': 'Unknown',
                'metrics': 'Not available',
                'timestamp': 'Unknown',
                'feature_count': 'Unknown'
            }
    
    def analyze_text_features(self, text):
        """Analyze various features of the input text"""
        if not text:
            return {}
        
        # Basic text statistics
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len(re.split(r'[.!?]+', text)) - 1
        
        # Sentiment analysis with TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Special character counts
        exclamation_count = text.count('!')
        question_count = text.count('?')
        uppercase_count = sum(1 for c in text if c.isupper())
        
        # Uppercase ratio
        uppercase_ratio = uppercase_count / char_count if char_count > 0 else 0
        
        # Average word length
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'avg_word_length': round(avg_word_length, 2),
            'polarity': round(polarity, 3),
            'subjectivity': round(subjectivity, 3),
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'uppercase_count': uppercase_count,
            'uppercase_ratio': round(uppercase_ratio, 3)
        }
    
    def get_prediction_explanation(self, text, top_features=10):
        """Get explanation for the prediction (top contributing features)"""
        if self.model is None or self.vectorizer is None:
            return "Model not loaded"
        
        # This is a simplified explanation - for more detailed explanations,
        # you might want to use libraries like SHAP or LIME
        
        processed_text = self.preprocess_single_text(text)
        text_vectorized = self.vectorizer.transform([processed_text])
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get the features that are present in this text
        feature_indices = text_vectorized.nonzero()[1]
        feature_scores = text_vectorized.toarray()[0]
        
        # Get model coefficients (only works for linear models)
        if hasattr(self.model, 'coef_'):
            coefficients = self.model.coef_[0]
            
            # Calculate feature importance for this prediction
            feature_importance = []
            for idx in feature_indices:
                if feature_scores[idx] > 0:
                    importance = feature_scores[idx] * coefficients[idx]
                    feature_importance.append((feature_names[idx], importance))
            
            # Sort by absolute importance
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            return feature_importance[:top_features]
        else:
            # For non-linear models, just return the most frequent features
            frequent_features = []
            for idx in feature_indices:
                if feature_scores[idx] > 0:
                    frequent_features.append((feature_names[idx], feature_scores[idx]))
            
            frequent_features.sort(key=lambda x: x[1], reverse=True)
            return frequent_features[:top_features]


def load_default_model():
    """Load the default trained model"""
    model_dir = "models"
    model_path = os.path.join(model_dir, "best_sentiment_model.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
    model_info_path = os.path.join(model_dir, "model_info.pkl")
    
    predictor = SentimentPredictor()
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        success = predictor.load_model(model_path, vectorizer_path, model_info_path)
        if success:
            return predictor
    
    return None


def predict_sentiment_simple(text):
    """Simple function for quick sentiment prediction"""
    predictor = load_default_model()
    
    if predictor is None:
        return {
            'sentiment': 'Unknown',
            'confidence': 0.0,
            'error': 'Model not found. Please train the model first.'
        }
    
    try:
        result = predictor.predict_sentiment(text)
        return result
    except Exception as e:
        return {
            'sentiment': 'Error',
            'confidence': 0.0,
            'error': str(e)
        }


def demo_predictions():
    """Demo function to test predictions"""
    # Sample texts for testing
    sample_texts = [
        "This movie was absolutely fantastic! Great acting and amazing storyline.",
        "Terrible movie. Waste of time and money. Very disappointed.",
        "The movie was okay, nothing special but not bad either.",
        "I loved every minute of it! Best film I've seen this year!",
        "Boring and predictable. The plot was weak and the acting was poor."
    ]
    
    predictor = load_default_model()
    
    if predictor is None:
        print("Error: Could not load model. Please train the model first.")
        return
    
    print("Sentiment Analysis Demo")
    print("=" * 50)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nSample {i}: {text}")
        result = predictor.predict_sentiment(text)
        
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Positive Probability: {result['positive_prob']:.3f}")
        print(f"Negative Probability: {result['negative_prob']:.3f}")
        
        # Show text features
        features = predictor.analyze_text_features(text)
        print(f"Text Features: {features}")
        print("-" * 40)


if __name__ == "__main__":
    demo_predictions()
