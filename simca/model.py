import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.stats
import joblib
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy.signal import savgol_filter
import math
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

class CorrectedScaler(StandardScaler):
    """StandardScaler with optional mean centering and scaling"""
    def __init__(self, with_mean=True, with_std=True):
        super().__init__(with_mean=with_mean, with_std=with_std)
        
class DD_SIMCA:
    """Data-Driven SIMCA model for single class modeling"""
    
    def __init__(self, n_components: int, alpha: float = 0.05):
        self.__n_components = n_components
        self.__alpha = alpha
        self.__is_fitted = False
        
    

    def fit(self, X_train: np.ndarray):
        """Fit the DD-SIMCA model to training data"""
        # 1. Autoscale X
        self.__ss = CorrectedScaler(with_mean=True, with_std=True)
        self.__X_train = X_train.copy()
        
        # Store original dimensions
        self.__n_samples, self.__J = X_train.shape
        
        # 2. Perform PCA on standardized coordinates
        self.__pca = PCA(n_components=self.__n_components)
        X_scaled = self.__ss.fit_transform(self.__X_train)
        self.__pca.fit(X_scaled)
        
        self.__is_fitted = True
        
        # 3. Compute critical distance
        # Compute OD (q) and SD (h) for training data
        h_vals, q_vals = self.h_q(self.__X_train)
        
        # Estimate scaling factors and degrees of freedom
        def dof(u0, u_vals):
            std = np.std(u_vals, ddof=1)
            if std < 1e-6:
                std = 1e-6
            val = 2.0 * u0**2 / std**2
            val = max(val, 1)
            val = min(val, 1000)  # cap to 1000
            return int(round(val))
        
        self.__h0, self.__q0 = np.mean(h_vals), np.mean(q_vals)
        self.__Nh, self.__Nq = dof(self.__h0, h_vals), dof(self.__q0, q_vals)
        # Robustly ensure they are plain Python ints and finite
        try:
            self.__Nh = int(float(self.__Nh))
            if not math.isfinite(self.__Nh):
                self.__Nh = 1
        except Exception:
            self.__Nh = 1
        try:
            self.__Nq = int(float(self.__Nq))
            if not math.isfinite(self.__Nq):
                self.__Nq = 1
        except Exception:
            self.__Nq = 1
        # Critical distance threshold
        df = self.__Nh + self.__Nq
        print(f"DEBUG: Nh={self.__Nh} (type {type(self.__Nh)}), Nq={self.__Nq} (type {type(self.__Nq)}), df={df} (type {type(df)})")
        if not isinstance(df, (int, float)) or not math.isfinite(df) or df <= 0:
            print("WARNING: Invalid degrees of freedom for chi2.ppf, setting df=1")
            df = 1
        self.__c_crit = scipy.stats.chi2.ppf(1.0 - self.__alpha, df)
        
        # Store training statistics for confidence calculation
        self.__train_distances = self.distance(self.__X_train)
        self.__max_train_distance = np.max(self.__train_distances)
        
        

    def h_q(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute score distance (h) and orthogonal distance (q)"""
        if not self.__is_fitted:
            raise ValueError("Model must be fitted before computing distances")
            
        X_std = self.__ss.transform(X)
        T = self.__pca.transform(X_std)
        X_pred = self.__pca.inverse_transform(T)
        
        # Orthogonal Distance (OD/q)
        q_vals = np.sum((X_std - X_pred)**2, axis=1)
        
        # Score Distance (SD/h)
        h_vals = np.sum(T**2 / self.__pca.explained_variance_, axis=1)
        
        return h_vals, q_vals

    def distance(self, X: np.ndarray) -> np.ndarray:
        """Compute combined distance metric"""
        h, q = self.h_q(X)
        return self.__Nh * h / self.__h0 + self.__Nq * q / self.__q0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class membership (True/False)"""
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        assert X.shape[1] == self.__J, f"Expected {self.__J} features, got {X.shape[1]}"
        
        # If c < c_crit, it belongs to the class
        distances = self.distance(X)
        return distances < self.__c_crit
    
    def confidence(self, X: np.ndarray) -> np.ndarray:
        """Calculate confidence scores (0-1, higher = more confident)"""
        distances = self.distance(X)
        # Normalize distances to [0, 1] range and invert (closer = higher confidence)
        max_distance = max(self.__c_crit * 2, self.__max_train_distance)
        normalized_distances = np.clip(distances / max_distance, 0, 1)
        return 1 - normalized_distances
    
    def get_model_info(self) -> Dict:
        """Get model information for interpretability"""
        if not self.__is_fitted:
            raise ValueError("Model must be fitted first")
            
        return {
            'n_components': self.__n_components,
            'explained_variance_ratio': self.__pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(self.__pca.explained_variance_ratio_),
            'critical_distance': self.__c_crit,
            'h0': self.__h0,
            'q0': self.__q0,
            'Nh': self.__Nh,
            'Nq': self.__Nq,
            'n_training_samples': self.__n_samples
        }



class MultiClassSIMCA:
    """
    Multi-class SIMCA classifier with automatic, per-class selection
    of the number of principal components (k) via cross-validation.
    """

    def __init__(
        self,
        n_components: Union[int, float, Dict[str, int]] = 0.95,
        alpha: float = 0.05,
        cv_max_k: int = 20,          # upper bound when CV-choosing k
        cv_folds: int = 5,
        random_state: int = 1,
    ):
        """
        Parameters
        ----------
        n_components : int | float | dict
            • int   – fixed number of PCs for **every** class  
            • float – cumulative variance threshold (e.g. 0.95) used as an
                       *upper* bound, then k is tuned by CV up to that limit  
            • dict  – explicit {class_label: k}
        alpha        : significance level for each DD-SIMCA model
        cv_max_k     : hard upper bound on PCs to try when using CV
        cv_folds     : number of StratifiedKFold splits in CV
        random_state : RNG seed for reproducibility
        """
        self.n_components   = n_components
        self.alpha          = alpha
        self.cv_max_k       = cv_max_k
        self.cv_folds       = cv_folds
        self.random_state   = random_state

        self.models: Dict[str, DD_SIMCA] = {}
        self.class_labels: np.ndarray    = np.array([])
        self.is_fitted    : bool         = False

    # ------------------------------------------------------------------
    # internal helper ---------------------------------------------------
    # ------------------------------------------------------------------
    def _best_k_for_class(
        self,
        X_cls: np.ndarray,
        class_label,
        X_all: np.ndarray,
        y_all: np.ndarray,
        k_upper: int,
    ) -> int:
        """
        Pick k by CV on a one-vs-rest problem for **one** class.

        Returns
        -------
        best_k : int   number of PCs that maximises accuracy
        """
        n_samples, n_feats = X_cls.shape
        k_upper = min(k_upper, self.cv_max_k, n_samples - 2, n_feats)

        if k_upper < 2:       # fallback if we have too few samples / feats
            return 2

        k_grid = range(2, k_upper + 1)
        best_k, best_acc = 2, 0.0

        # build a balanced “rest” set the same size as X_cls
        X_rest_full = X_all[y_all != class_label]
        rest_idx    = np.random.RandomState(self.random_state).choice(
            len(X_rest_full), size=len(X_cls), replace=False
        )
        X_rest = X_rest_full[rest_idx]

        X_cv = np.vstack([X_cls, X_rest])
        y_cv = np.hstack([np.ones(len(X_cls)), np.zeros(len(X_rest))])

        skf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        for k in k_grid:
            preds = np.zeros_like(y_cv)
            for train, test in skf.split(X_cv, y_cv):
                model_tmp = DD_SIMCA(n_components=k, alpha=self.alpha)
                # fit only on POSITIVE-class samples in this fold
                model_tmp.fit(X_cv[train][y_cv[train] == 1])
                preds[test] = model_tmp.predict(X_cv[test])
            acc = accuracy_score(y_cv, preds)
            if acc > best_acc:
                best_acc, best_k = acc, k

        return best_k
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit a DD-SIMCA model for each mineral class.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self.class_labels = np.unique(y)

        for class_label in self.class_labels:
            X_cls = X[y == class_label]

            # --- decide how many PCs for this class --------------------
            if isinstance(self.n_components, dict):
                k = self.n_components.get(class_label, 5)

            elif isinstance(self.n_components, int):
                k = self.n_components

            elif isinstance(self.n_components, float):
                # use variance threshold as *upper* bound, then CV-tune ≤ that
                pca_tmp = PCA().fit(X_cls)
                k_upper = (
                    np.searchsorted(
                        np.cumsum(pca_tmp.explained_variance_ratio_),
                        self.n_components,
                    )
                    + 1
                )
                k = self._best_k_for_class(
                    X_cls,
                    class_label,
                    X_all=X,
                    y_all=y,
                    k_upper=k_upper,
                )
            else:
                raise ValueError("Unsupported n_components type")

            # safety clip
            k = int(max(2, min(k, X_cls.shape[0] - 2, X_cls.shape[1])))

            # -----------------------------------------------------------
            model = DD_SIMCA(n_components=k, alpha=self.alpha)
            model.fit(X_cls)
            self.models[class_label] = model

        self.is_fitted = True
        return self  # enable chaining
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class labels and confidence scores
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test spectra
            
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class labels or 'Unknown' for outliers
        confidences : array, shape (n_samples,)
            Confidence scores for predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        n_samples = X.shape[0]
        predictions = []
        confidences = []
        
        for i in range(n_samples):
            sample = X[i].reshape(1, -1)
            
            # Test against all class models
            class_memberships = {}
            class_confidences = {}
            class_distances = {}
            
            for class_label in self.class_labels:
                membership = self.models[class_label].predict(sample)[0]
                confidence = self.models[class_label].confidence(sample)[0]
                distance = self.models[class_label].distance(sample)[0]
                
                class_memberships[class_label] = membership
                class_confidences[class_label] = confidence
                class_distances[class_label] = distance
            
            # Determine final prediction
            accepted_classes = [cls for cls, member in class_memberships.items() if member]
            
            if len(accepted_classes) == 0:
                # No class accepts this sample - it's an outlier
                predictions.append('Unknown')
                confidences.append(0.0)
            elif len(accepted_classes) == 1:
                # Exactly one class accepts - clear prediction
                pred_class = accepted_classes[0]
                predictions.append(pred_class)
                confidences.append(class_confidences[pred_class])
            else:
                # Multiple classes accept - choose the one with highest confidence
                best_class = max(accepted_classes, key=lambda cls: class_confidences[cls])
                predictions.append(best_class)
                confidences.append(class_confidences[best_class])
                
        return np.array(predictions), np.array(confidences)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Returns:
        --------
        probabilities : array, shape (n_samples, n_classes)
            Probability estimates for each class
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        n_samples = X.shape[0]
        n_classes = len(self.class_labels)
        probabilities = np.zeros((n_samples, n_classes))
        
        for i, class_label in enumerate(self.class_labels):
            confidences = self.models[class_label].confidence(X)
            memberships = self.models[class_label].predict(X)
            
            # Set probability to confidence if accepted, otherwise 0
            probabilities[:, i] = confidences * memberships
            
        # Normalize probabilities (if any class accepts the sample)
        row_sums = probabilities.sum(axis=1)
        non_zero_rows = row_sums > 0
        probabilities[non_zero_rows] = probabilities[non_zero_rows] / row_sums[non_zero_rows].reshape(-1, 1)
        
        return probabilities
    
    def get_model_info(self) -> Dict:
        """Get information about all fitted models"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        info = {
            'n_classes': len(self.class_labels),
            'class_labels': self.class_labels,
            'alpha': self.alpha,
            'class_models': {}
        }
        
        for class_label in self.class_labels:
            info['class_models'][class_label] = self.models[class_label].get_model_info()
            
        return info
    
    def plot_residuals(self, X: np.ndarray, y: np.ndarray = None, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot residual distances for interpretability
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test spectra
        y : array-like, shape (n_samples,), optional
            True labels for comparison
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
            
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        n_samples = X.shape[0]
        n_classes = len(self.class_labels)
        
        # Calculate distances for all samples and classes
        distances = np.zeros((n_samples, n_classes))
        thresholds = np.zeros(n_classes)
        
        for i, class_label in enumerate(self.class_labels):
            distances[:, i] = self.models[class_label].distance(X)
            thresholds[i] = self.models[class_label].get_model_info()['critical_distance']
        
        # Create subplot
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Distance heatmap
        im = axes[0, 0].imshow(distances.T, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Distance to Each Class Model')
        axes[0, 0].set_xlabel('Sample Index')
        axes[0, 0].set_ylabel('Class')
        axes[0, 0].set_yticks(range(n_classes))
        axes[0, 0].set_yticklabels(self.class_labels)
        plt.colorbar(im, ax=axes[0, 0])
        
        # Plot 2: Threshold comparison
        for i, class_label in enumerate(self.class_labels):
            axes[0, 1].scatter(range(n_samples), distances[:, i], 
                             label=f'{class_label}', alpha=0.6, s=20)
            axes[0, 1].axhline(y=thresholds[i], color=plt.cm.tab10(i), 
                             linestyle='--', alpha=0.8)
        
        axes[0, 1].set_title('Distances vs Thresholds')
        axes[0, 1].set_xlabel('Sample Index')
        axes[0, 1].set_ylabel('Distance')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 3: Prediction visualization
        predictions, confidences = self.predict(X)
        
        # Color code by prediction
        colors = []
        for pred in predictions:
            if pred == 'Unknown':
                colors.append('red')
            else:
                idx = list(self.class_labels).index(pred)
                colors.append(plt.cm.tab10(idx))
        
        scatter = axes[1, 0].scatter(range(n_samples), confidences, 
                                   c=colors, alpha=0.7, s=50)
        axes[1, 0].set_title('Prediction Confidence')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Confidence')
        axes[1, 0].set_ylim(0, 1)
        
        # Plot 4: Confusion matrix (if true labels provided)
        if y is not None:
            y_true = np.array(y)
            unique_labels = np.unique(np.concatenate([y_true, predictions]))
            
            # Create confusion matrix
            conf_matrix = np.zeros((len(unique_labels), len(unique_labels)))
            for i, true_label in enumerate(unique_labels):
                for j, pred_label in enumerate(unique_labels):
                    conf_matrix[i, j] = np.sum((y_true == true_label) & (predictions == pred_label))
            
            im = axes[1, 1].imshow(conf_matrix, cmap='Blues')
            axes[1, 1].set_title('Confusion Matrix')
            axes[1, 1].set_xlabel('Predicted')
            axes[1, 1].set_ylabel('True')
            axes[1, 1].set_xticks(range(len(unique_labels)))
            axes[1, 1].set_yticks(range(len(unique_labels)))
            axes[1, 1].set_xticklabels(unique_labels, rotation=45)
            axes[1, 1].set_yticklabels(unique_labels)
            
            # Add text annotations
            for i in range(len(unique_labels)):
                for j in range(len(unique_labels)):
                    text = axes[1, 1].text(j, i, int(conf_matrix[i, j]),
                                         ha="center", va="center", color="black")
        else:
            axes[1, 1].text(0.5, 0.5, 'No true labels\nprovided', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig("simca_residuals.png", dpi=300, bbox_inches="tight")
        plt.show()
        