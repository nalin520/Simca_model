## Overview

This project implements a robust pipeline for mineral classification using the SIMCA (Soft Independent Modeling of Class Analogy) algorithm, leveraging spectral data from the RRUFF database. The pipeline is designed for flexible feature engineering, preprocessing, and model evaluation.

## Pipeline Flow

The main analysis is orchestrated by the `run_complete_rruff_simca_analysis` function, which is invoked via `main.py`. The pipeline consists of the following steps:

1. Loading Data
   - Loads a pickle file containing mineral spectral data (typically a pandas DataFrame or dictionary).
   - Explores and prints the structure of the loaded data.

2. Exploring Data Structure
   - Provides a detailed exploration of the data, including columns, types, and sample values.

3. Extracting Spectra and Labels
   - Extracts spectral data and mineral labels from the loaded dataset.
   - Handles both DataFrame and dictionary-based data structures.

4. Preprocessing Data
   - Applies smoothing (Savitzky–Golay), normalization (area, z-score, or min-max), and outlier removal.
   - Filters rare classes and retains only the top N most common mineral classes.
   - Splits the data into training and test sets.

5. Feature Engineering
   - Constructs a feature matrix using a variety of spectral features:
     - Derivative features
     - Peak statistics
     - Region ratios
     - PCA features
     - Entropy
     - Percentiles
     - Statistical moments
   - Handles missing or non-finite values robustly.

6. SIMCA Model Training and Hyperparameter Tuning
   - Performs a grid search over SIMCA model parameters to find the best configuration.
   - Trains the model on the training set and evaluates on the test set.

7. Evaluation
   - Reports overall and per-class accuracy.
   - Displays a confusion matrix for the top confused class pairs.

8. Saving Model and Metadata
   - Saves the trained SIMCA model, metadata, and feature engineering parameters for future use.

9. Pipeline Completion
   - Prints a summary and instructions for loading the model and metadata later.

## Example Usage

The pipeline is typically run from the command line. The command used in the sample output is:

```
python main.py --pickle data/rruff.pkl --mineral_column Name --spectrum_columns Data
```

### Key Arguments
- `--pickle`: Path to the input pickle file containing the spectral data.
- `--mineral_column`: Name of the column containing mineral labels (for DataFrame input).
- `--spectrum_columns`: Name or pattern for the column(s) containing spectral data.
- Additional flags allow customization of feature engineering, normalization, and filtering.

## Sample Output

The pipeline provides detailed, step-by-step output, including:
- Data loading and structure exploration
- Extraction and preprocessing statistics
- Feature engineering diagnostics
- Model training, hyperparameter search, and evaluation metrics
- Per-class accuracy and confusion matrix
- Paths to saved model, metadata, and feature engineering parameters

See `sample_result2.txt` for a full example of the pipeline's output, including the command used, data statistics, and model performance.

## Output Files
- `simca_model_<timestamp>.joblib`: Trained SIMCA model
- `simca_metadata_<timestamp>.json`: Model metadata and training statistics
- `feature_engineering_params_<timestamp>.json`: Parameters for feature engineering and preprocessing

## Loading the Model Later
To load and use the trained model and metadata:

```
import joblib
import json
model = joblib.load('simca_model_<timestamp>.joblib')
with open('simca_metadata_<timestamp>.json', 'r') as f:
    metadata = json.load(f)
```

