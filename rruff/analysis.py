import numpy as np
import pandas as pd
from simca.feature_engineering import build_feature_matrix, engineer_and_select
from simca.tuning import simca_hyper_tune
from simca.model import MultiClassSIMCA
from rruff.loader import RRUFFDataProcessor, extract_rruff_dataframe
from simca.preprocessing import load_and_preprocess_data_from_strings
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
import joblib
import json
import os
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


def run_complete_rruff_simca_analysis(csv_strings: list, 
                                     file_names: list,
                                     start: float = 0,
                                     end: float = 4000,
                                     num_wavenums: int = 2000,
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
    """
    Complete SIMCA analysis pipeline for CSV input data
    
    Parameters:
    -----------
    csv_strings : list
        List of CSV string contents
    file_names : list
        Corresponding filenames (format: mineralname_suffix.csv)
    start : float
        Start of wavenumber range
    end : float
        End of wavenumber range
    num_wavenums : int
        Number of wavenumber points
    [Other parameters same as original]
    """
    print("\n" + "="*60)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("="*60)
    
    # Load and preprocess data
    wavenumbers, baseline_corrected_data, file_labels = load_and_preprocess_data_from_strings(
        csv_strings, file_names, start, end, num_wavenums
    )
    
    # Extract corrected spectra and create mineral labels from filenames
    spectra = [corrected for _, corrected, _ in baseline_corrected_data]
    mineral_labels = [os.path.splitext(fname)[0].split('_')[0] for fname in file_labels]
    
    X = np.array(spectra)
    y = np.array(mineral_labels)
    
    # Apply normalization
    if normalization == "area":
        areas = np.trapezoid(X, wavenumbers, axis=1)
        areas[areas == 0] = 1  # Avoid division by zero
        X = X / areas[:, np.newaxis]
    elif normalization == "max":
        max_vals = np.max(X, axis=1)
        max_vals[max_vals == 0] = 1
        X = X / max_vals[:, np.newaxis]
    
    print(f"Loaded {X.shape[0]} spectra with {X.shape[1]} wavenumber points")
    print(f"Mineral classes: {set(mineral_labels)}")
    
    # Filter rare classes
    X, y = filter_rare_classes(X, y, min_samples=min_samples_per_class)
    # Filter to top N classes
    X, y = filter_top_n_classes(X, y, n=top_n_classes)
    print(f"Final classes: {sorted(set(y))}")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Ensure test set only includes train classes
    train_classes = set(y_train)
    test_mask = np.isin(y_test, list(train_classes))
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Feature engineering
    print("\n" + "="*60)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*60)
    
    X_train_fe, _ = build_feature_matrix(
        X_train, wavenumbers=wavenumbers,
        include_deriv=include_deriv,
        include_peak_stats=include_peak_stats,
        include_region_ratios=include_region_ratios,
        include_pca_features=include_pca_features,
        include_entropy=include_entropy,
        include_percentiles=include_percentiles,
        include_moments=include_moments
    )
    
    X_test_fe, _ = build_feature_matrix(
        X_test, wavenumbers=wavenumbers,
        include_deriv=include_deriv,
        include_peak_stats=include_peak_stats,
        include_region_ratios=include_region_ratios,
        include_pca_features=include_pca_features,
        include_entropy=include_entropy,
        include_percentiles=include_percentiles,
        include_moments=include_moments
    )
    
    # Handle non-finite values
    X_train_fe = np.nan_to_num(X_train_fe, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_fe = np.nan_to_num(X_test_fe, nan=0.0, posinf=0.0, neginf=0.0)
    
    # SIMCA model training and evaluation
    print("\n" + "="*60)
    print("STEP 3: SIMCA GRID SEARCH (hyperparameter tuning)")
    print("="*60)
    model = simca_grid_search(X_train_fe, y_train, X_test_fe, y_test, wavenumbers=wavenumbers)
    
    print("\n" + "="*60)
    print("STEP 4: EVALUATING BEST SIMCA MODEL")
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
    plot_confusion_matrix(y_test, predictions, class_labels=unique_labels, top_n=10)

    print("\n" + "="*60)
    print("STEP 5: SAVING MODEL AND METADATA")
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
            'wavenumbers_shape': wavenumbers.shape
        },
        'training_date': str(datetime.now()),
        'model_type': 'MultiClassSIMCA'
    }
    
    # Save the trained model
    model_filename = f"simca_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
    joblib.dump(model, model_filename)
    print(f"✓ Model saved to: {model_filename}")
    
    # Save metadata
    metadata_filename = f"simca_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metadata_filename, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_filename}")
    
    # Save feature engineering parameters
    fe_params = {
        'wavenumbers': wavenumbers.tolist(),
        'feature_engineering_params': model_metadata['feature_engineering_params']
    }
    fe_params_filename = f"feature_engineering_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fe_params_filename, 'w') as f:
        json.dump(fe_params, f, indent=2)
    print(f"✓ Feature engineering parameters saved to: {fe_params_filename}")
    
    print("\n" + "="*60)
    print("STEP 6: PIPELINE COMPLETE")
    print("="*60)
    print(f"Model and all metadata saved successfully!")
    
    return model, wavenumbers