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
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def savgol_smooth(y, window=15, polyorder=3):
    """Savitzky-Golay smoothing with safety checks."""
    window = min(len(y)//2*2+1, window)          # force odd window ≤ len(y)
    if window < polyorder + 2:
        return y
    return savgol_filter(y, window, polyorder)

def median_MAD_scale(y, eps=1e-9):
    """Robust per-spectrum scaling: (y - median) / MAD."""
    med = np.median(y)
    mad = np.median(np.abs(y - med)) + eps
    return (y - med) / mad

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
            return int(np.max([round(2.0 * u0**2 / np.std(u_vals, ddof=1) ** 2, 0), 1]))
        
        self.__h0, self.__q0 = np.mean(h_vals), np.mean(q_vals)
        self.__Nh, self.__Nq = dof(self.__h0, h_vals), dof(self.__q0, q_vals)
        
        # Critical distance threshold
        self.__c_crit = scipy.stats.chi2.ppf(1.0 - self.__alpha, self.__Nh + self.__Nq)
        
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
            • int   - fixed number of PCs for **every** class  
            • float - cumulative variance threshold (e.g. 0.95) used as an
                       *upper* bound, then k is tuned by CV up to that limit  
            • dict  - explicit {class_label: k}
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
        
class RRUFFDataProcessor:
    """
    Processor for RRUFF mineral spectral data
    """
    
    def __init__(self, pickle_path: str):
        """
        Initialize with path to pickle file
        
        Parameters:
        -----------
        pickle_path : str
            Path to the RRUFF pickle file
        """
        self.pickle_path = pickle_path
        self.raw_data = None
        self.processed_data = None
        self.wavenumbers = None
        self.spectra = None
        self.labels = None
        self.mineral_names = None
        
    def load_data(self):
        """Load the pickle file and explore its structure"""
        print(f"Loading pickle file: {self.pickle_path}")
        
        try:
            with open(self.pickle_path, 'rb') as f:
                self.raw_data = pickle.load(f)
            
            print("✓ Pickle file loaded successfully!")
            print(f"Data type: {type(self.raw_data)}")
            
            # Explore the structure
            if isinstance(self.raw_data, dict):
                print(f"Dictionary keys: {list(self.raw_data.keys())}")
                for key, value in self.raw_data.items():
                    print(f"  {key}: {type(value)} - {np.array(value).shape if hasattr(value, '__len__') else 'scalar'}")
            
            elif isinstance(self.raw_data, (list, tuple)):
                print(f"Length: {len(self.raw_data)}")
                if len(self.raw_data) > 0:
                    print(f"First element type: {type(self.raw_data[0])}")
                    if hasattr(self.raw_data[0], 'shape'):
                        print(f"First element shape: {self.raw_data[0].shape}")
            
            elif isinstance(self.raw_data, np.ndarray):
                print(f"Array shape: {self.raw_data.shape}")
                print(f"Array dtype: {self.raw_data.dtype}")
            
            elif isinstance(self.raw_data, pd.DataFrame):
                print(f"DataFrame shape: {self.raw_data.shape}")
                print(f"Columns: {list(self.raw_data.columns)}")
                print(f"Data types:\n{self.raw_data.dtypes}")
                
            return True
            
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            return False
    
    def explore_data_structure(self):
        """Provide detailed exploration of the data structure"""
        if self.raw_data is None:
            print("Please load data first using load_data()")
            return
        
        print("\n" + "="*50)
        print("DETAILED DATA EXPLORATION")
        print("="*50)
        
        def explore_recursive(obj, name="root", max_depth=3, current_depth=0):
            if current_depth > max_depth:
                return
            
            indent = "  " * current_depth
            
            if isinstance(obj, dict):
                print(f"{indent}{name}: dict with {len(obj)} keys")
                for key, value in list(obj.items())[:5]:  # Show first 5 keys
                    explore_recursive(value, f"{name}['{key}']", max_depth, current_depth + 1)
                if len(obj) > 5:
                    print(f"{indent}  ... and {len(obj) - 5} more keys")
            
            elif isinstance(obj, (list, tuple)):
                print(f"{indent}{name}: {type(obj).__name__} with {len(obj)} elements")
                if len(obj) > 0:
                    explore_recursive(obj[0], f"{name}[0]", max_depth, current_depth + 1)
                    if len(obj) > 1:
                        print(f"{indent}  ... and {len(obj) - 1} more elements")
            
            elif isinstance(obj, np.ndarray):
                print(f"{indent}{name}: numpy array {obj.shape}, dtype: {obj.dtype}")
                if obj.ndim <= 2 and obj.size < 20:
                    print(f"{indent}  Sample values: {obj.flatten()[:10]}")
            
            elif isinstance(obj, pd.DataFrame):
                print(f"{indent}{name}: pandas DataFrame {obj.shape}")
                print(f"{indent}  Columns: {list(obj.columns)}")
                
            else:
                print(f"{indent}{name}: {type(obj).__name__}")
                if hasattr(obj, '__len__') and len(str(obj)) < 100:
                    print(f"{indent}  Value: {obj}")
        
        explore_recursive(self.raw_data)
    
    def extract_spectra_and_labels(self, 
                                  spectra_key: str = None,
                                  labels_key: str = None,
                                  wavenumbers_key: str = None,
                                  mineral_column: str = None,
                                  spectrum_columns: str = None):
        """
        Extract spectra and labels from the loaded data
        
        Parameters:
        -----------
        spectra_key : str, optional
            Key for spectral data in dictionary
        labels_key : str, optional
            Key for mineral labels in dictionary
        wavenumbers_key : str, optional
            Key for wavenumber data
        mineral_column : str, optional
            Column name for mineral labels (if DataFrame)
        spectrum_columns : str or list, optional
            Column names or pattern for spectral data (if DataFrame)
        """
        
        if self.raw_data is None:
            print("Please load data first using load_data()")
            return False
        
        try:
            # Handle different data structures
            if isinstance(self.raw_data, dict):
                # Dictionary-based data
                print("Processing dictionary-based data...")
                
                # Try to automatically detect keys if not provided
                if spectra_key is None:
                    possible_spectra_keys = ['spectra', 'intensities', 'data', 'X', 'raman_spectra', 'spectra_data']
                    for key in possible_spectra_keys:
                        if key in self.raw_data:
                            spectra_key = key
                            break
                
                if labels_key is None:
                    possible_label_keys = ['labels', 'minerals', 'classes', 'y', 'mineral_names', 'targets']
                    for key in possible_label_keys:
                        if key in self.raw_data:
                            labels_key = key
                            break
                
                if wavenumbers_key is None:
                    possible_wave_keys = ['wavenumbers', 'wavelengths', 'frequencies', 'x_axis', 'raman_shift']
                    for key in possible_wave_keys:
                        if key in self.raw_data:
                            wavenumbers_key = key
                            break
                
                print(f"Using keys - Spectra: {spectra_key}, Labels: {labels_key}, Wavenumbers: {wavenumbers_key}")
                
                # Extract data
                if spectra_key and spectra_key in self.raw_data:
                    self.spectra = np.array(self.raw_data[spectra_key])
                    print(f"✓ Extracted spectra: {self.spectra.shape}")
                
                if labels_key and labels_key in self.raw_data:
                    self.labels = np.array(self.raw_data[labels_key])
                    print(f"✓ Extracted labels: {len(self.labels)} samples")
                
                if wavenumbers_key and wavenumbers_key in self.raw_data:
                    self.wavenumbers = np.array(self.raw_data[wavenumbers_key])
                    print(f"✓ Extracted wavenumbers: {len(self.wavenumbers)} points")
            
            elif isinstance(self.raw_data, pd.DataFrame):
                # DataFrame-based data
                print("Processing DataFrame-based data...")
                df = self.raw_data
                
                # Extract mineral labels
                if mineral_column and mineral_column in df.columns:
                    self.labels = df[mineral_column].values
                    print(f"✓ Extracted labels from column '{mineral_column}': {len(self.labels)} samples")
                
                # Extract spectral data
                if spectrum_columns:
                    if isinstance(spectrum_columns, str):
                        # Pattern matching for column names
                        spectral_cols = [col for col in df.columns if spectrum_columns in col]
                    else:
                        spectral_cols = spectrum_columns
                    
                    if spectral_cols:
                        self.spectra = df[spectral_cols].values
                        print(f"✓ Extracted spectra from {len(spectral_cols)} columns: {self.spectra.shape}")
                        
                        # Try to extract wavenumbers from column names
                        try:
                            self.wavenumbers = np.array([float(col.split('_')[-1]) for col in spectral_cols])
                            print(f"✓ Extracted wavenumbers from column names: {len(self.wavenumbers)} points")
                        except:
                            print("Could not extract wavenumbers from column names")
            
            # Validate extracted data
            if self.spectra is not None and self.labels is not None:
                if len(self.spectra) != len(self.labels):
                    print(f"WARNING: Spectra ({len(self.spectra)}) and labels ({len(self.labels)}) have different lengths!")
                    # Try to match them
                    min_len = min(len(self.spectra), len(self.labels))
                    self.spectra = self.spectra[:min_len]
                    self.labels = self.labels[:min_len]
                    print(f"Truncated to {min_len} samples")
                
                # Get unique minerals
                self.mineral_names = np.unique(self.labels)
                print(f"✓ Found {len(self.mineral_names)} unique minerals: {self.mineral_names}")
                
                # Print sample distribution
                unique, counts = np.unique(self.labels, return_counts=True)
                print("\nSample distribution:")
                for mineral, count in zip(unique, counts):
                    print(f"  {mineral}: {count} samples")
                
                return True
            
            else:
                print("Could not extract both spectra and labels. Please specify the correct keys/columns.")
                return False
                
        except Exception as e:
            print(f"Error extracting data: {e}")
            return False
    
    def preprocess_data(self, 
                       wavenumber_range: Tuple[float, float] = None,
                       apply_savgol: bool = True,   
                       use_derivative: bool = True,
                       normalize: bool = True,
                       remove_outliers: bool = True,
                       min_samples_per_class: int = 5):
        """
        Preprocess the extracted spectral data
        
        Parameters:
        -----------
        wavenumber_range : tuple, optional
            (min, max) wavenumber range to keep
        normalize : bool
            Whether to normalize spectra
        remove_outliers : bool
            Whether to remove outlier spectra
        min_samples_per_class : int
            Minimum samples required per mineral class
        """
        
        # ------------------------------------------------------------------
        # 1) COPY DATA
        # ------------------------------------------------------------------
        spectra = self.spectra.copy()
        labels  = self.labels.copy()
        wavenumbers = self.wavenumbers.copy() if self.wavenumbers is not None else None

        # ------------------------------------------------------------------
        # 2) OPTIONAL WAVENUMBER WINDOW
        # ------------------------------------------------------------------
        if wavenumber_range is not None and wavenumbers is not None:
            lo, hi = wavenumber_range

            # make sure both are NumPy arrays so boolean indexing is valid
            spectra     = np.asarray(spectra)
            wavenumbers = np.asarray(wavenumbers)

            keep = (wavenumbers >= lo) & (wavenumbers <= hi)
            if keep.sum() == 0:
                raise ValueError(
                    f"No channels remain in the {lo}-{hi} cm⁻¹ window; "
                    "check the requested range."
                )

            spectra     = spectra[:, keep]
            wavenumbers = wavenumbers[keep]
            print(f"✓ Filtered {lo}-{hi} cm⁻¹ → {spectra.shape[1]} points")
       
        # ------------------------------------------------------------------
        # 3) OPTIONAL SAVITZKY–GOLAY SMOOTHING
        # ------------------------------------------------------------------
        if apply_savgol:
            win, order = 15, 3                # feel free to tune
            spectra = np.asarray([
                savgol_smooth(row, win, order)         # <- uses the helper you defined
                for row in spectra
            ])
            print(f"✓ Savitzky–Golay smoothing applied (win={win}, poly={order})")
        
        if use_derivative:
            # SG derivative keeps noise low and aligns grid spacing
            deriv = np.asarray([
                savgol_filter(row, 15, 3, deriv=1) for row in spectra
            ])
            spectra = np.hstack([spectra, deriv])      # concat orig + dI/dν
            if wavenumbers is not None:
                wavenumbers = np.hstack([wavenumbers, wavenumbers])  # dummy duplicate
            print("✓ First-derivative features appended")

        # ------------------------------------------------------------------
        # 4) VARIANCE FILTER – DROP FLATTEST 20 % OF CHANNELS
        # ------------------------------------------------------------------
        var = np.var(spectra, axis=0)
        mask_var = var > np.percentile(var, 40)
        spectra  = spectra[:, mask_var]
        if wavenumbers is not None:
            wavenumbers = wavenumbers[mask_var]
        print(f"✓ Variance filter kept {spectra.shape[1]} informative channels")

        # ------------------------------------------------------------------
        # 5) CLASS PRUNING (min_samples_per_class)
        # ------------------------------------------------------------------
        unique, counts = np.unique(labels, return_counts=True)
        keep_classes   = unique[counts >= min_samples_per_class]
        if len(keep_classes) < len(unique):
            keep_mask = np.isin(labels, keep_classes)
            spectra, labels = spectra[keep_mask], labels[keep_mask]
            print(f"✓ Removed classes with <{min_samples_per_class} samples")

        # ------------------------------------------------------------------
        # 6) OPTIONAL OUTLIER REMOVAL (same logic as before, but on new spectra)
        # ------------------------------------------------------------------
        if remove_outliers:
            median_spec = np.median(spectra, axis=0)
            mad_vec     = np.median(np.abs(spectra - median_spec), axis=1)
            thr         = np.median(mad_vec) + 3 * np.std(mad_vec)
            ok          = mad_vec < thr
            spectra, labels = spectra[ok], labels[ok]
            print(f"✓ Removed {np.sum(~ok)} spectral outliers")

        # ------------------------------------------------------------------
        # 7) SAVE BACK
        # ------------------------------------------------------------------
        self.processed_data = {
            'spectra'      : spectra,
            'labels'       : labels,
            'wavenumbers'  : wavenumbers,
            'mineral_names': np.unique(labels)
        }
        print("\n✓ Preprocessing complete!")
        print(f"Final dataset: {spectra.shape[0]} samples, {spectra.shape[1]} features")
        print(f"Classes: {len(self.processed_data['mineral_names'])}")
        return True

    
    def get_training_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        Split processed data into training and testing sets
        
        Parameters:
        -----------
        test_size : float
            Fraction of data to use for testing
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        X_train, X_test, y_train, y_test : arrays
            Training and testing data
        """
        
        if self.processed_data is None:
            print("Please preprocess data first")
            return None
        
        X = self.processed_data['spectra']
        y = self.processed_data['labels']
        
        # Stratified split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y
        )
        
        print(f"✓ Split data: {X_train.shape[0]} training, {X_test.shape[0]} testing samples")
        
        return X_train, X_test, y_train, y_test
    
    def plot_sample_spectra(self, n_samples: int = 5, figsize: Tuple[int, int] = (12, 8)):
        """Plot sample spectra for each mineral class"""
        
        if self.processed_data is None:
            print("Please preprocess data first")
            return
        
        spectra = self.processed_data['spectra']
        labels = self.processed_data['labels']
        wavenumbers = self.processed_data['wavenumbers']
        mineral_names = self.processed_data['mineral_names']
        
        n_classes = len(mineral_names)
        n_cols = min(3, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        x_axis = wavenumbers if wavenumbers is not None else range(spectra.shape[1])
        
        for i, mineral in enumerate(mineral_names):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            # Get samples for this mineral
            mineral_mask = labels == mineral
            mineral_spectra = spectra[mineral_mask]
            
            # Plot up to n_samples
            n_plot = min(n_samples, len(mineral_spectra))
            for j in range(n_plot):
                ax.plot(x_axis, mineral_spectra[j], alpha=0.7, linewidth=1)
            
            ax.set_title(f'{mineral} (n={len(mineral_spectra)})')
            ax.set_xlabel('Wavenumber (cm⁻¹)' if wavenumbers is not None else 'Feature Index')
            ax.set_ylabel('Intensity')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_classes, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig("sample_spectra.png", dpi=300, bbox_inches="tight")
        plt.show()
        


def train_simca_on_rruff_data(pickle_path: str, 
                             spectra_key: str = None,
                             labels_key: str = None,
                             wavenumbers_key: str = None,
                             n_components: int = 5,
                             alpha: float = 0.01):
    """
    Complete pipeline to train SIMCA model on RRUFF data
    
    Parameters:
    -----------
    pickle_path : str
        Path to RRUFF pickle file
    spectra_key : str, optional
        Key for spectral data in pickle file
    labels_key : str, optional
        Key for mineral labels in pickle file
    wavenumbers_key : str, optional
        Key for wavenumber data in pickle file
    n_components : int
        Number of PCA components for SIMCA
    alpha : float
        Significance level for outlier detection
        
    Returns:
    --------
    trained_model : MultiClassSIMCA
        Trained SIMCA model
    processor : RRUFFDataProcessor
        Data processor with loaded data
    """
    
    # Initialize processor
    processor = RRUFFDataProcessor(pickle_path)
    
    # Load and explore data
    if not processor.load_data():
        return None, None
    
    # Show data structure
    processor.explore_data_structure()
    
    # Extract spectra and labels
    print("\n" + "="*50)
    print("EXTRACTING SPECTRA AND LABELS")
    print("="*50)
    
    if not processor.extract_spectra_and_labels(
        spectra_key=spectra_key,
        labels_key=labels_key,
        wavenumbers_key=wavenumbers_key
    ):
        return None, None
    
    # Preprocess data
    if not processor.preprocess_data():
        return None, None
    
    # Get training data
    X_train, X_test, y_train, y_test = processor.get_training_data()
    
    # Plot sample spectra
    print("\nPlotting sample spectra...")
    processor.plot_sample_spectra()
    
    # Train SIMCA model
    print("\n" + "="*50)
    print("TRAINING SIMCA MODEL")
    print("="*50)
    
    model = MultiClassSIMCA(n_components=n_components, alpha=alpha)
    model.fit(X_train, y_train)
    
    # Test the model
    print("Testing model...")
    predictions, confidences = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Test accuracy: {accuracy:.3f}")
    
    # Show detailed results
    print("\nDetailed results:")
    unique_labels = np.unique(y_test)
    for label in unique_labels:
        mask = y_test == label
        label_accuracy = np.mean(predictions[mask] == y_test[mask])
        print(f"  {label}: {label_accuracy:.3f} ({np.sum(mask)} samples)")
    
    # Plot results
    model.plot_residuals(X_test, y_test)
    
    return model, processor

def extract_rruff_dataframe(processor):
    """
    Extract spectral data from RRUFF DataFrame format
    """
    df = processor.raw_data
    
    print("Extracting from RRUFF DataFrame...")
    
    # Extract mineral names as labels
    processor.labels = df['Name'].values
    print(f"✓ Extracted {len(processor.labels)} mineral labels")
    
    # Examine the data format
    sample_data = df['Data'].iloc[0]
    print(f"Sample data type: {type(sample_data)}")
    
    # Initialize lists to store extracted data
    all_spectra = []
    all_wavenumbers = []
    valid_indices = []
    
    print("Processing spectral data...")
    
    for i, data in enumerate(df['Data']):
        try:
            if isinstance(data, pd.DataFrame):
                # Use the first two columns, whatever they’re called
                wavenumbers = data.iloc[:, 0].to_numpy(dtype=float)
                intensities = data.iloc[:, 1].to_numpy(dtype=float)
            elif isinstance(data, np.ndarray):
                
                if data.ndim == 2 and data.shape[1] == 2:
                    # Format: [wavenumber, intensity] pairs
                    wavenumbers = data[:, 0]
                    intensities = data[:, 1]
                elif data.ndim == 2 and data.shape[0] == 2:
                    # Format: [[wavenumbers], [intensities]]
                    wavenumbers = data[0, :]
                    intensities = data[1, :]
                elif data.ndim == 1:
                    # Format: intensity values only
                    intensities = data
                    wavenumbers = np.arange(len(intensities))  # Default indexing
                else:
                    print(f"Unexpected data format at index {i}: {data.shape}")
                    continue
            elif isinstance(data, (list, tuple)) and len(data) == 2:
                # Format: [wavenumbers, intensities]
                wavenumbers = np.array(data[0])
                intensities = np.array(data[1])
            else:
                print(f"Unknown data format at index {i}: {type(data)}")
                continue
            
            # Validate the data
            if len(wavenumbers) != len(intensities):
                print(f"Mismatch at index {i}: wavenumbers {len(wavenumbers)} vs intensities {len(intensities)}")
                continue
            
            # Remove NaN and infinite values
            valid_mask = np.isfinite(intensities) & np.isfinite(wavenumbers)
            if not np.any(valid_mask):
                continue
                
            wavenumbers = wavenumbers[valid_mask]
            intensities = intensities[valid_mask]
            
            all_spectra.append(intensities)
            all_wavenumbers.append(wavenumbers)
            valid_indices.append(i)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"✓ Successfully processed {len(all_spectra)} spectra")
    
    # Find common wavenumber range
    if len(all_wavenumbers) > 0:
        # Find the intersection of all wavenumber ranges
        min_wave = max(waves.min() for waves in all_wavenumbers)
        max_wave = min(waves.max() for waves in all_wavenumbers)
        
        print(f"Common wavenumber range: {min_wave:.1f} - {max_wave:.1f} cm⁻¹")
        # --- choose a sensible resolution ------------------------------------
        res_list = [np.median(np.diff(w)) for w in all_wavenumbers if len(w) > 1]

        # use the 10th-percentile spacing so very fine outliers don’t dominate
        resolution = np.percentile(res_list, 10)

        # never go below 0.5 cm⁻¹
        resolution = max(resolution, 0.5)

        # if the grid would still be huge, loosen it further
        max_points_allowed = 50_000          # pick any limit you like
        n_points = int(np.floor((max_wave - min_wave) / resolution)) + 1
        if n_points > max_points_allowed:
            resolution = (max_wave - min_wave) / max_points_allowed
            n_points = max_points_allowed + 1

        # now build the grid
        common_wavenumbers = np.linspace(min_wave, max_wave, n_points)
        
        
        print(f"Interpolating to common grid: {len(common_wavenumbers)} points")
        
        # Interpolate all spectra to common grid
        interpolated_spectra = []
        for i, (waves, intensities) in enumerate(zip(all_wavenumbers, all_spectra)):
            try:
                # Sort by wavenumber (in case they're not sorted)
                sort_idx = np.argsort(waves)
                waves = waves[sort_idx]
                intensities = intensities[sort_idx]
                
                # Interpolate to common grid
                interp_intensities = np.interp(common_wavenumbers, waves, intensities)
                interpolated_spectra.append(interp_intensities)
                
            except Exception as e:
                print(f"Error interpolating spectrum {i}: {e}")
                continue
        
        processor.spectra = np.array(interpolated_spectra)
        processor.wavenumbers = common_wavenumbers
        processor.labels = processor.labels[valid_indices]
        
        print(f"✓ Final dataset: {processor.spectra.shape}")
        print(f"✓ Wavenumber range: {processor.wavenumbers.min():.1f} - {processor.wavenumbers.max():.1f} cm⁻¹")
        
        # Get unique minerals
        processor.mineral_names = np.unique(processor.labels)
        print(f"✓ Found {len(processor.mineral_names)} unique minerals")
        
        # Print sample distribution
        unique, counts = np.unique(processor.labels, return_counts=True)
        print("\nTop 10 minerals by sample count:")
        sorted_idx = np.argsort(counts)[::-1]
        for i in range(min(10, len(unique))):
            idx = sorted_idx[i]
            print(f"  {unique[idx]}: {counts[idx]} samples")
        
        return True
    
    else:
        print("No valid spectral data found")
        return False

def run_complete_rruff_simca_analysis(pickle_path):
    """
    Complete pipeline for RRUFF SIMCA analysis
    """
    
    # Step 1: Initialize processor and load data
    print("="*60)
    print("STEP 1: LOADING RRUFF DATA")
    print("="*60)
    
    processor = RRUFFDataProcessor(pickle_path)
    if not processor.load_data():
        print("Failed to load data!")
        return None, None
    
    # Step 2: Analyze data structure (run the analysis code first)
    print("\n" + "="*60)
    print("STEP 2: ANALYZING DATA STRUCTURE")
    print("="*60)
    
    # Examine the first few samples
    print("First 5 sample names:")
    print(processor.raw_data['Name'].head().tolist())
    
    first_data = processor.raw_data['Data'].iloc[0]
    print(f"\nFirst sample data type: {type(first_data)}")

    if isinstance(first_data, pd.DataFrame):
        print(f"Shape: {first_data.shape}")
        # show the first five wavenumber–intensity pairs instead of flattening
        print("First few (wavenumber, intensity) rows:")
        print(first_data.head())
    else:
        # assume array-like
        print(f"Shape: {first_data.shape}")
        print(f"First few values: {np.asarray(first_data).ravel()[:10]}")

    
    # Step 3: Extract spectral data using custom function
    print("\n" + "="*60)
    print("STEP 3: EXTRACTING SPECTRAL DATA")
    print("="*60)
    
    if not extract_rruff_dataframe(processor):
        print("Failed to extract spectral data!")
        return None, None
    
    # Step 4: Preprocess the data
    print("\n" + "="*60)
    print("STEP 4: PREPROCESSING DATA")
    print("="*60)
    
    # Filter to common minerals with enough samples
    unique_labels, counts = np.unique(processor.labels, return_counts=True)
    min_samples = 10  # Minimum samples per mineral
    
    print(f"Minerals with ≥{min_samples} samples:")
    sufficient_minerals = unique_labels[counts >= min_samples]
    sufficient_counts = counts[counts >= min_samples]
    
    for mineral, count in zip(sufficient_minerals, sufficient_counts):
        print(f"  {mineral}: {count} samples")
    
    # Filter dataset to include only minerals with sufficient samples
    mask = np.isin(processor.labels, sufficient_minerals)
    processor.spectra = processor.spectra[mask]
    processor.labels = processor.labels[mask]
    
    print(f"\nFiltered dataset: {processor.spectra.shape[0]} samples, {len(sufficient_minerals)} minerals")
    
    # Further preprocessing
    processor.processed_data = {
        'spectra': processor.spectra,
        'labels': processor.labels,
        'wavenumbers': processor.wavenumbers,
        'mineral_names': sufficient_minerals
    }
    
    # Apply basic preprocessing
    if not processor.preprocess_data(
        normalize=True,
        remove_outliers=True,
        min_samples_per_class=min_samples
    ):
        print("Preprocessing failed!")
        return None, None
    
    norms = np.linalg.norm(processor.spectra, axis=1)
    keep  = norms > np.percentile(norms, 10)
    processor.spectra, processor.labels = processor.spectra[keep], processor.labels[keep]
    print(f"✓ Removed lowest-quality 10 % spectra, {processor.spectra.shape[0]} remain")
    # Step 5: Split data
    print("\n" + "="*60)
    print("STEP 5: SPLITTING DATA")
    print("="*60)
    
    X_train, X_test, y_train, y_test = processor.get_training_data(test_size=0.2, random_state=42)
    
    # Step 6: Train SIMCA model
    print("\n" + "="*60)
    print("STEP 6: TRAINING SIMCA MODEL")
    print("="*60)
    
    # Start with fewer components for faster training
    n_components = 0.95
    model = MultiClassSIMCA(n_components=n_components, alpha=0.01)
    
    print(f"Training SIMCA with {n_components} components...")
    model.fit(X_train, y_train)
    
    # Step 7: Evaluate model
    print("\n" + "="*60)
    print("STEP 7: EVALUATING MODEL")
    print("="*60)
    
    # Make predictions
    predictions, confidences = model.predict(X_test)
    
    # Calculate overall accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Overall test accuracy: {accuracy:.3f}")
    
    # Calculate per-class accuracy
    print("\nPer-class accuracy:")
    unique_test_labels = np.unique(y_test)
    for label in unique_test_labels:
        mask = y_test == label
        if np.sum(mask) > 0:
            label_acc = np.mean(predictions[mask] == label)
            n_samples = np.sum(mask)
            print(f"  {label}: {label_acc:.3f} ({n_samples} samples)")
    
    # Count unknown predictions
    unknown_count = np.sum(predictions == 'Unknown')
    print(f"\nUnknown predictions: {unknown_count}/{len(predictions)} ({unknown_count/len(predictions)*100:.1f}%)")
    
    # Step 8: Model diagnostics
    print("\n" + "="*60)
    print("STEP 8: MODEL DIAGNOSTICS")
    print("="*60)
    
    model_info = model.get_model_info()
    print(f"Number of classes: {model_info['n_classes']}")
    print(f"Alpha (significance level): {model_info['alpha']}")
    
    for class_label in model_info['class_labels']:
        class_info = model_info['class_models'][class_label]
        print(f"\n{class_label}:")
        print(f"  Components: {class_info['n_components']}")
        print(f"  Explained variance: {class_info['cumulative_variance_ratio'][-1]:.3f}")
        print(f"  Training samples: {class_info['n_training_samples']}")
    
    # Step 9: Visualize results
    print("\n" + "="*60)
    print("STEP 9: VISUALIZING RESULTS")
    print("="*60)
    
    # Plot some sample spectra
    processor.plot_sample_spectra(n_samples=3)
    
    # Plot model residuals
    model.plot_residuals(X_test, y_test)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"✓ Trained SIMCA model with {n_components} components")
    print(f"✓ {len(model_info['class_labels'])} mineral classes")
    print(f"✓ {accuracy:.1%} test accuracy")
    print(f"✓ {X_train.shape[0]} training samples")
    print(f"✓ {X_test.shape[0]} test samples")
    
    return model, processor

# Run the complete analysis
if __name__ == "__main__":
    pickle_path = "/home/ubuntu/SIMCA/data/rruff.pkl"
    
    # Run the complete workflow
    model, processor = run_complete_rruff_simca_analysis(pickle_path)
    
    if model is not None:
        print("\n" + "="*60)
        print("SUCCESS! Your SIMCA model is ready to use.")
        print("="*60)
        
        # Example: Make predictions on new data
        # new_predictions, new_confidences = model.predict(new_spectra)
        
        joblib.dump(model, 'rruff_simca_model.pkl')
        print("Model saved as 'rruff_simca_model.pkl'")
    else:
        print("Failed to create SIMCA model. Check the error messages above.")
