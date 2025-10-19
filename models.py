"""
Machine Learning Models Module
"""

import numpy as np
import joblib
from tensorflow.keras.models import load_model

# =============================================================================
# MODEL MANAGER CLASS
# =============================================================================

class ModelManager:
    """Manages all machine learning models."""
    
    def __init__(self):
        """Initialize model manager and load models."""
        self.cnn_model = None
        self.svm_model = None
        self.svm_scaler = None
        self.rf_model = None
        self._load_models()
    
    def _load_models(self):
        """Load all machine learning models."""
        try:
            self.cnn_model = load_model('mnist_cnn_model.keras')
            print("CNN model loaded successfully")
        except Exception as e:
            print(f"Failed to load CNN model: {e}")
            self.cnn_model = None
        
        try:
            self.svm_model = joblib.load('mnist_svm_model.joblib')
            self.svm_scaler = joblib.load('mnist_svm_scaler.joblib')
            print("SVM model and scaler loaded successfully")
        except Exception as e:
            print(f"Failed to load SVM model: {e}")
            self.svm_model = None
            self.svm_scaler = None
        
        try:
            self.rf_model = joblib.load('mnist_rf_model.joblib')
            print("RF model loaded successfully")
        except Exception as e:
            print(f"Failed to load RF model: {e}")
            self.rf_model = None
    
    def get_loaded_models(self):
        """Get list of loaded models."""
        loaded = []
        if self.cnn_model is not None:
            loaded.append("CNN")
        if self.svm_model is not None:
            loaded.append("SVM")
        if self.rf_model is not None:
            loaded.append("RF")
        return loaded if loaded else ["None"]
    
    def predict_all_models(self, chips):
        """
        Get predictions from all available models.
        
        Args:
            chips: List of 28x28 digit chips
        
        Returns:
            Tuple of (all_predictions, all_confidences)
        """
        if not chips:
            return {}, {}
        
        # Prepare data
        X = np.array(chips).reshape(len(chips), -1)
        
        all_predictions = {}
        all_confidences = {}
        
        # CNN predictions
        if self.cnn_model is not None:
            cnn_pred, cnn_conf = self._predict_cnn(chips)
            all_predictions['CNN'] = cnn_pred
            all_confidences['CNN'] = cnn_conf
        
        # SVM predictions
        if self.svm_model is not None:
            svm_pred, svm_conf = self._predict_svm(X)
            all_predictions['SVM'] = svm_pred
            all_confidences['SVM'] = svm_conf
        
        # RF predictions
        if self.rf_model is not None:
            rf_pred, rf_conf = self._predict_rf(X)
            all_predictions['RF'] = rf_pred
            all_confidences['RF'] = rf_conf
        
        return all_predictions, all_confidences
    
    def predict_with_model(self, chips, model_name="auto"):
        """
        Predict using specified model.
        
        Args:
            chips: List of 28x28 digit chips
            model_name: Model to use ("auto", "cnn", "svm", "rf")
        
        Returns:
            Tuple of (predictions, all_predictions, all_confidences)
        """
        if not chips:
            return [], {}, {}
        
        # Get all model predictions
        all_predictions, all_confidences = self.predict_all_models(chips)
        
        if model_name == "auto":
            # Filter out models with empty predictions
            valid_models = {k: v for k, v in all_predictions.items() if len(v) > 0}
            if not valid_models:
                return [], all_predictions, all_confidences
            
            # Pick the model with highest confidence among valid models
            best_model = max(valid_models.keys(), key=lambda k: all_confidences[k])
            return all_predictions[best_model], all_predictions, all_confidences
        
        elif model_name == "cnn":
            return all_predictions.get('CNN', []), all_predictions, all_confidences
        
        elif model_name == "svm":
            return all_predictions.get('SVM', []), all_predictions, all_confidences
        
        elif model_name == "rf":
            return all_predictions.get('RF', []), all_predictions, all_confidences
        
        return [], all_predictions, all_confidences
    
    def _predict_cnn(self, chips):
        """Predict using CNN model."""
        try:
            X = np.array(chips).reshape(len(chips), 28, 28, 1)
            predictions = self.cnn_model.predict(X, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            confidences = np.max(predictions, axis=1)
            avg_confidence = np.mean(confidences)
            return predicted_classes.tolist(), avg_confidence
        except Exception as e:
            print(f"CNN prediction error: {e}")
            return [], 0.0
    
    def _predict_svm(self, X):
        """Predict using SVM model."""
        try:
            X_scaled = self.svm_scaler.transform(X)
            predictions = self.svm_model.predict(X_scaled)
            
            # Try predict_proba first (if model was trained with probability=True)
            try:
                proba = self.svm_model.predict_proba(X_scaled)
                confidences = np.max(proba, axis=1)
                avg_confidence = np.mean(confidences)
            except AttributeError:
                # Fallback to decision_function if predict_proba not available
                decision_scores = self.svm_model.decision_function(X_scaled)
                if len(decision_scores.shape) > 1:
                    decision_scores = np.max(decision_scores, axis=1)
                # Normalize decision scores to 0-1 range using sigmoid
                confidences = 1 / (1 + np.exp(-np.abs(decision_scores)))
                avg_confidence = np.mean(confidences)
            
            return predictions.tolist(), avg_confidence
        except Exception as e:
            print(f"SVM prediction error: {e}")
            return [], 0.0
    
    def _predict_rf(self, X):
        """Predict using Random Forest model."""
        try:
            predictions = self.rf_model.predict(X)
            confidences = self.rf_model.predict_proba(X)
            avg_confidence = np.mean(np.max(confidences, axis=1))
            return predictions.tolist(), avg_confidence
        except Exception as e:
            print(f"RF prediction error: {e}")
            return [], 0.0
