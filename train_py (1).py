"""
Model training and evaluation for sentiment analysis
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
import mlflow
import mlflow.sklearn
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing functions
from preprocess import load_and_preprocess_data, create_feature_vectors, add_advanced_features


class SentimentModelTrainer:
    """
    A comprehensive class for training and evaluating sentiment analysis models
    """
    
    def __init__(self, data_path, use_mlflow=False):
        self.data_path = data_path
        self.use_mlflow = use_mlflow
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        
        # Initialize MLflow if requested
        if use_mlflow:
            mlflow.set_experiment("sentiment_analysis")
    
    def load_data(self, test_size=0.2, random_state=42):
        """Load and preprocess the data"""
        print("Loading and preprocessing data...")
        self.X_train, self.X_test, self.y_train, self.y_test, self.df_processed = load_and_preprocess_data(
            self.data_path, test_size=test_size, random_state=random_state
        )
        
        # Create feature vectors
        self.X_train_vec, self.X_test_vec, self.vectorizer = create_feature_vectors(
            self.X_train, self.X_test, method='tfidf', max_features=10000
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        print(f"Feature matrix shape: {self.X_train_vec.shape}")
    
    def initialize_models(self):
        """Initialize different models for comparison"""
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(random_state=42, probability=True),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
        }
    
    def train_single_model(self, name, model, X_train, X_test, y_train, y_test):
        """Train and evaluate a single model"""
        print(f"Training {name}...")
        
        if self.use_mlflow:
            with mlflow.start_run(run_name=name):
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
                
                # Log metrics to MLflow
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model
                mlflow.sklearn.log_model(model, f"model_{name.lower().replace(' ', '_')}")
        else:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        return model, metrics, y_pred, y_pred_proba
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def train_all_models(self):
        """Train all models and compare performance"""
        self.initialize_models()
        
        for name, model in tqdm(self.models.items(), desc="Training models"):
            trained_model, metrics, y_pred, y_pred_proba = self.train_single_model(
                name, model, self.X_train_vec, self.X_test_vec, self.y_train, self.y_test
            )
            
            self.results[name] = {
                'model': trained_model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Track best model
            if metrics['f1_score'] > self.best_score:
                self.best_score = metrics['f1_score']
                self.best_model = name
        
        print(f"\nBest model: {self.best_model} (F1 Score: {self.best_score:.4f})")
    
    def hyperparameter_tuning(self, model_name='Logistic Regression'):
        """Perform hyperparameter tuning for the specified model"""
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        if model_name == 'Logistic Regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        elif model_name == 'Random Forest':
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        elif model_name == 'SVM':
            model = SVC(random_state=42, probability=True)
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train_vec, self.y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Evaluate on test set
        y_pred = best_model.predict(self.X_test_vec)
        y_pred_proba = best_model.predict_proba(self.X_test_vec)[:, 1]
        metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        print(f"Test set performance: {metrics}")
        
        return best_model, grid_search.best_params_, metrics
    
    def cross_validate_model(self, model_name='Logistic Regression', cv=5):
        """Perform cross-validation for a specific model"""
        model = self.models[model_name]
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, self.X_train_vec, self.y_train, cv=cv, scoring='f1')
        
        print(f"Cross-validation scores for {model_name}:")
        print(f"Mean F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def plot_results(self):
        """Plot model comparison results"""
        # Prepare data for plotting
        model_names = list(self.results.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(metrics_names):
            if i < len(axes):
                values = [self.results[model][metrics][metric] for model in model_names 
                         if metric in self.results[model]['metrics']]
                valid_models = [model for model in model_names 
                              if metric in self.results[model]['metrics']]
                
                axes[i].bar(valid_models, values, color='skyblue', alpha=0.7)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_ylabel('Score')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].set_ylim(0, 1)
                
                # Add value labels on bars
                for j, v in enumerate(values):
                    axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Plot confusion matrix for best model
        if len(axes) > 5:
            best_model_results = self.results[self.best_model]
            cm = confusion_matrix(self.y_test, best_model_results['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[5])
            axes[5].set_title(f'Confusion Matrix - {self.best_model}')
            axes[5].set_xlabel('Predicted')
            axes[5].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for name, results in self.results.items():
            if results['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, results['probabilities'])
                auc_score = results['metrics']['roc_auc']
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def save_best_model(self, model_dir='models'):
        """Save the best performing model"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        best_model_obj = self.results[self.best_model]['model']
        
        # Save model
        model_path = os.path.join(model_dir, 'best_sentiment_model.pkl')
        joblib.dump(best_model_obj, model_path)
        
        # Save vectorizer
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Save model info
        model_info = {
            'model_name': self.best_model,
            'metrics': self.results[self.best_model]['metrics'],
            'timestamp': datetime.now().isoformat(),
            'feature_count': self.X_train_vec.shape[1]
        }
        
        info_path = os.path.join(model_dir, 'model_info.pkl')
        joblib.dump(model_info, info_path)
        
        print(f"Best model ({self.best_model}) saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
        print(f"Model info saved to {info_path}")
        
        return model_path, vectorizer_path, info_path
    
    def generate_report(self):
        """Generate a comprehensive training report"""
        print("\n" + "="*60)
        print("SENTIMENT ANALYSIS MODEL TRAINING REPORT")
        print("="*60)
        
        print(f"\nDataset Information:")
        print(f"- Total samples: {len(self.df_processed)}")
        print(f"- Training samples: {len(self.X_train)}")
        print(f"- Test samples: {len(self.X_test)}")
        print(f"- Feature dimensions: {self.X_train_vec.shape[1]}")
        
        print(f"\nClass Distribution:")
        class_dist = pd.Series(self.y_train).value_counts()
        for class_label, count in class_dist.items():
            sentiment = 'Positive' if class_label == 1 else 'Negative'
            print(f"- {sentiment}: {count} ({count/len(self.y_train)*100:.1f}%)")
        
        print(f"\nModel Performance Comparison:")
        print("-" * 80)
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
        print("-" * 80)
        
        for name, results in self.results.items():
            metrics = results['metrics']
            print(f"{name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} "
                  f"{metrics.get('roc_auc', 'N/A'):<10.4s}")
        
        print("-" * 80)
        print(f"\nBest Model: {self.best_model}")
        print(f"Best F1-Score: {self.best_score:.4f}")
        
        # Classification report for best model
        print(f"\nDetailed Classification Report for {self.best_model}:")
        print("-" * 50)
        best_predictions = self.results[self.best_model]['predictions']
        print(classification_report(self.y_test, best_predictions, 
                                  target_names=['Negative', 'Positive']))


def main():
    """Main training pipeline"""
    # Configuration
    DATA_PATH = "/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"
    USE_MLFLOW = False  # Set to True if you want to use MLflow tracking
    
    # Initialize trainer
    trainer = SentimentModelTrainer(DATA_PATH, use_mlflow=USE_MLFLOW)
    
    # Load and preprocess data
    trainer.load_data()
    
    # Train all models
    trainer.train_all_models()
    
    # Generate report
    trainer.generate_report()
    
    # Plot results
    trainer.plot_results()
    trainer.plot_roc_curves()
    
    # Hyperparameter tuning for best model
    print(f"\nPerforming hyperparameter tuning for the best model...")
    tuned_model, best_params, tuned_metrics = trainer.hyperparameter_tuning(trainer.best_model)
    
    # Cross-validation
    print(f"\nPerforming cross-validation...")
    cv_scores = trainer.cross_validate_model(trainer.best_model)
    
    # Save the best model
    model_paths = trainer.save_best_model()
    
    print("\nTraining completed successfully!")
    return trainer


if __name__ == "__main__":
    trainer = main()