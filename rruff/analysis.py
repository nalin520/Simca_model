import numpy as np
import pandas as pd
from simca.feature_engineering import build_feature_matrix, engineer_and_select
from simca.tuning import simca_hyper_tune
from simca.model import MultiClassSIMCA
from rruff.loader import RRUFFDataProcessor, extract_rruff_dataframe
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
import joblib
import json
from datetime import datetime

def plot_confusion_matrix(y_true, y_pred, class_labels, top_n=10):
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    # Find most confused pairs
    cm_offdiag = cm.copy()
    np.fill_diagonal(cm_offdiag, 0)
    flat = cm_offdiag.flatten()
    top_idx = np.argpartition(flat, -top_n)[-top_n:]
    for idx in top_idx[np.argsort(-flat[top_idx])]:
        i, j = divmod(idx, len(class_labels))
        if cm_offdiag[i, j] > 0:
            print(f"Most confused: {class_labels[i]} → {class_labels[j]}: {cm_offdiag[i, j]} times")

def filter_rare_classes(X, y, min_samples=10):
    unique, counts = np.unique(y, return_counts=True)
    keep_classes = unique[counts >= min_samples]
    mask = np.isin(y, keep_classes)
    return X[mask], y[mask]

def filter_top_n_classes(X, y, n=10):
    top_classes = [c for c, _ in Counter(y).most_common(n)]
    mask = np.isin(y, top_classes)
    return X[mask], y[mask]

def simca_grid_search(X_train, y_train, X_test, y_test, wavenumbers=None):
    best_acc = 0
    best_params = None
    best_model = None
    # Expanded grid search for better tuning
    for n_components in [0.995]:
        for alpha in [0.005]:
            print(f"Grid search: n_components={n_components}, alpha={alpha}")
            model = MultiClassSIMCA(n_components=n_components, alpha=alpha)
            print("  Fitting model...")
            model.fit(X_train, y_train)
            print("  Predicting...")
            preds, _ = model.predict(X_test)
            acc = np.mean(preds == y_test)
            print(f"  acc={acc:.3f}")
            if acc > best_acc:
                best_acc = acc
                best_params = (n_components, alpha)
                best_model = model
    if best_params is None:
        raise RuntimeError("No valid SIMCA parameters found during grid search. Check your data and parameter ranges.")
    print(f"Best SIMCA params: n_components={best_params[0]}, alpha={best_params[1]} (acc={best_acc:.3f})")
    return best_model

def train_simca_on_rruff_data(pickle_path: str, 
                             spectra_key: str = None,
                             labels_key: str = None,
                             wavenumbers_key: str = None,
                             n_components: int = 0.95,
                             alpha: float = 0.05,
                             mineral_column: str = None,
                             spectrum_columns: list = None,
                             min_samples_per_class: int = 10):
    processor = RRUFFDataProcessor(pickle_path)
    if not processor.load_data():
        return None, None
    processor.explore_data_structure()
    print("\n" + "="*50)
    print("EXTRACTING SPECTRA AND LABELS")
    print("="*50)
    if not processor.extract_spectra_and_labels(
        spectra_key=spectra_key,
        labels_key=labels_key,
        wavenumbers_key=wavenumbers_key,
        mineral_column=mineral_column,
        spectrum_columns=spectrum_columns
    ):
        return None, None
    if not processor.preprocess_data():
        return None, None
    X_train, X_test, y_train, y_test = processor.get_training_data()
    # Feature engineering
    X_train_fe, _ = build_feature_matrix(X_train, wavenumbers=processor.wavenumbers)
    X_test_fe, _ = build_feature_matrix(X_test, wavenumbers=processor.wavenumbers)
    # Sanitize feature matrices
    if not np.all(np.isfinite(X_train_fe)):
        print("WARNING: X_train_fe contains non-finite values. They will be replaced with 0.")
    if not np.all(np.isfinite(X_test_fe)):
        print("WARNING: X_test_fe contains non-finite values. They will be replaced with 0.")
    X_train_fe = np.nan_to_num(X_train_fe, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_fe = np.nan_to_num(X_test_fe, nan=0.0, posinf=0.0, neginf=0.0)
    # Filter rare classes
    X_train_fe, y_train = filter_rare_classes(X_train_fe, y_train, min_samples=min_samples_per_class)
    X_test_fe, y_test = filter_rare_classes(X_test_fe, y_test, min_samples=min_samples_per_class)
    # Hyperparameter tuning
    print("\n" + "="*50)
    print("SIMCA GRID SEARCH (hyperparameter tuning)")
    print("="*50)
    model = simca_grid_search(X_train_fe, y_train, X_test_fe, y_test, wavenumbers=processor.wavenumbers)
    # Test the best model
    print("\n" + "="*50)
    print("EVALUATING BEST SIMCA MODEL")
    print("="*50)
    predictions, confidences = model.predict(X_test_fe)
    accuracy = np.mean(predictions == y_test)
    print(f"Test accuracy: {accuracy:.3f}")
    print("\nPer-class accuracy:")
    unique_labels = np.unique(y_test)
    for label in unique_labels:
        mask = y_test == label
        label_accuracy = np.mean(predictions[mask] == y_test[mask])
        print(f"  {label}: {label_accuracy:.3f} ({np.sum(mask)} samples)")
    # Confusion matrix
    print("\nConfusion matrix (top confused pairs):")
    plot_confusion_matrix(y_test, predictions, class_labels=unique_labels, top_n=10)
    return model, processor

def run_complete_rruff_simca_analysis(pickle_path, 
                             spectra_key: str = None,
                             labels_key: str = None,
                             wavenumbers_key: str = None,
                             n_components: int = 0.95,
                             alpha: float = 0.05,
                             mineral_column: str = None,
                             spectrum_columns: list = None,
                             min_samples_per_class: int = 10,
                             top_n_classes: int = 10,
                             include_deriv: bool = True,
                             include_peak_stats: bool = True,
                             include_region_ratios: bool = True,
                             include_pca_features: bool = True,
                             include_entropy: bool = True,
                             include_percentiles: bool = True,
                             include_moments: bool = True,
                             normalization: str = "area",
                             outlier_percentile: float = 10.0):
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    processor = RRUFFDataProcessor(pickle_path)
    if not processor.load_data():
        print("Failed to load data!")
        return None, None
    print("\n" + "="*60)
    print("STEP 2: EXPLORING DATA STRUCTURE")
    print("="*60)
    processor.explore_data_structure()
    print("\n" + "="*60)
    print("STEP 3: EXTRACTING SPECTRA AND LABELS")
    print("="*60)
    if not processor.extract_spectra_and_labels(
        spectra_key=spectra_key,
        labels_key=labels_key,
        wavenumbers_key=wavenumbers_key,
        mineral_column=mineral_column,
        spectrum_columns=spectrum_columns
    ):
        print("Failed to extract spectra and labels!")
        return None, None
    print("\n" + "="*60)
    print("STEP 4: PREPROCESSING DATA")
    print("="*60)
    if not processor.preprocess_data(normalization=normalization, outlier_percentile=outlier_percentile):
        print("Preprocessing failed!")
        return None, None
    # After preprocessing, get full spectra and labels
    spectra = processor.processed_data['spectra']
    labels = processor.processed_data['labels']
    # 1. Filter rare classes on full data
    spectra, labels = filter_rare_classes(spectra, labels, min_samples=min_samples_per_class)
    # 2. Filter to top N classes on full data
    spectra, labels = filter_top_n_classes(spectra, labels, n=top_n_classes)
    print(f"Classes used: {sorted(set(labels))}")
    # 3. Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    # 4. Filter test set to only include classes present in train set
    train_classes = set(y_train)
    test_mask = np.isin(y_test, list(train_classes))
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    # 5. Feature engineering
    print("\n" + "="*60)
    print("STEP 5: FEATURE ENGINEERING")
    print("="*60)
    X_train_fe, _ = build_feature_matrix(
        X_train, wavenumbers=processor.wavenumbers,
        include_deriv=include_deriv,
        include_peak_stats=include_peak_stats,
        include_region_ratios=include_region_ratios,
        include_pca_features=include_pca_features,
        include_entropy=include_entropy,
        include_percentiles=include_percentiles,
        include_moments=include_moments
    )
    X_test_fe, _ = build_feature_matrix(
        X_test, wavenumbers=processor.wavenumbers,
        include_deriv=include_deriv,
        include_peak_stats=include_peak_stats,
        include_region_ratios=include_region_ratios,
        include_pca_features=include_pca_features,
        include_entropy=include_entropy,
        include_percentiles=include_percentiles,
        include_moments=include_moments
    )
    # Sanitize feature matrices
    if not np.all(np.isfinite(X_train_fe)):
        print("WARNING: X_train_fe contains non-finite values. They will be replaced with 0.")
    if not np.all(np.isfinite(X_test_fe)):
        print("WARNING: X_test_fe contains non-finite values. They will be replaced with 0.")
    X_train_fe = np.nan_to_num(X_train_fe, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_fe = np.nan_to_num(X_test_fe, nan=0.0, posinf=0.0, neginf=0.0)
    # Diagnostics and grid search as before
    print("\n" + "="*60)
    print("STEP 8: SIMCA GRID SEARCH (hyperparameter tuning)")
    print("="*60)
    print(f"X_train_fe shape: {X_train_fe.shape}, y_train shape: {y_train.shape}")
    print(f"X_test_fe shape: {X_test_fe.shape}, y_test shape: {y_test.shape}")
    print(f"Unique y_train: {np.unique(y_train)}")
    print(f"Unique y_test: {np.unique(y_test)}")
    print(f"Any NaN in X_train_fe? {np.isnan(X_train_fe).any()}")
    print(f"Any NaN in X_test_fe? {np.isnan(X_test_fe).any()}")
    print(f"Any NaN in y_train? {np.isnan(y_train).any() if y_train.dtype.kind == 'f' else 'N/A'}")
    print(f"Any NaN in y_test? {np.isnan(y_test).any() if y_test.dtype.kind == 'f' else 'N/A'}")
    if X_train_fe.shape[0] == 0 or X_test_fe.shape[0] == 0:
        raise RuntimeError("Train or test set is empty after filtering. Relax min_samples_per_class or top_n_classes.")
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise RuntimeError("Fewer than 2 classes remain after filtering. Relax min_samples_per_class or top_n_classes.")
    model = simca_grid_search(X_train_fe, y_train, X_test_fe, y_test, wavenumbers=processor.wavenumbers)
    print("\n" + "="*60)
    print("STEP 9: EVALUATING BEST SIMCA MODEL")
    print("="*60)
    predictions, confidences = model.predict(X_test_fe)
    accuracy = np.mean(predictions == y_test)
    print(f"Test accuracy: {accuracy:.3f}")
    print("\nPer-class accuracy:")
    unique_labels = np.unique(y_test)
    for label in unique_labels:
        mask = y_test == label
        label_accuracy = np.mean(predictions[mask] == y_test[mask])
        print(f"  {label}: {label_accuracy:.3f} ({np.sum(mask)} samples)")
    print("\nConfusion matrix (top confused pairs):")
    if len(unique_labels) > 15:
        plt.figure()
        plot_confusion_matrix(y_test, predictions, class_labels=unique_labels, top_n=10)
        plt.savefig("confusion_matrix.png")
        print("Confusion matrix saved to confusion_matrix.png")
    else:
        plot_confusion_matrix(y_test, predictions, class_labels=unique_labels, top_n=10)
    print("\n" + "="*60)
    print("STEP 10: SAVING MODEL AND METADATA")
    print("="*60)
    
    # Create model metadata
    model_metadata = {
        'accuracy': float(accuracy),
        'n_classes': len(unique_labels),
        'classes': unique_labels.tolist(),
        'feature_engineering_params': {
            'include_deriv': include_deriv,
            'include_peak_stats': include_peak_stats,
            'include_region_ratios': include_region_ratios,
            'include_pca_features': include_pca_features,
            'include_entropy': include_entropy,
            'include_percentiles': include_percentiles,
            'include_moments': include_moments
        },
        'preprocessing_params': {
            'normalization': normalization,
            'outlier_percentile': outlier_percentile,
            'min_samples_per_class': min_samples_per_class,
            'top_n_classes': top_n_classes
        },
        'data_info': {
            'train_samples': X_train_fe.shape[0],
            'test_samples': X_test_fe.shape[0],
            'n_features': X_train_fe.shape[1],
            'wavenumbers_shape': processor.wavenumbers.shape if processor.wavenumbers is not None else None
        },
        'training_date': str(datetime.now()),
        'model_type': 'MultiClassSIMCA'
    }
    
    # Save the trained model using joblib (efficient for scikit-learn compatible objects)
    model_filename = f"simca_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
    joblib.dump(model, model_filename)
    print(f"✓ Model saved to: {model_filename}")
    
    # Save metadata as JSON
    metadata_filename = f"simca_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metadata_filename, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_filename}")
    
    # Save feature engineering parameters for preprocessing new data
    fe_params = {
        'wavenumbers': processor.wavenumbers.tolist() if processor.wavenumbers is not None else None,
        'feature_engineering_params': model_metadata['feature_engineering_params']
    }
    fe_params_filename = f"feature_engineering_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fe_params_filename, 'w') as f:
        json.dump(fe_params, f, indent=2)
    print(f"✓ Feature engineering parameters saved to: {fe_params_filename}")
    
    print("\n" + "="*60)
    print("STEP 11: PIPELINE COMPLETE")
    print("="*60)
    print(f"Model and all metadata saved successfully!")
    print(f"To load and use the model later:")
    print(f"  model = joblib.load('{model_filename}')")
    print(f"  with open('{metadata_filename}', 'r') as f:")
    print(f"      metadata = json.load(f)")
    return model, processor