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
from simca.preprocessing import savgol_smooth
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

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
                        
                        # If only one column and entries are DataFrames, extract intensity arrays
                        if len(spectral_cols) == 1:
                            sample = self.spectra[0, 0]
                            if isinstance(sample, pd.DataFrame):
                                print("Detected DataFrame objects in spectra column; extracting intensity arrays...")
                                # Extract wavenumbers and intensities for each sample
                                all_wavenumbers = []
                                all_intensities = []
                                for row in self.spectra:
                                    df = row[0]
                                    if isinstance(df, pd.DataFrame):
                                        wavenumbers = df.iloc[:, 0].to_numpy(dtype=float)
                                        intensities = df.iloc[:, 1].to_numpy(dtype=float)
                                        all_wavenumbers.append(wavenumbers)
                                        all_intensities.append(intensities)
                                # Find common grid
                                min_wave = max(waves.min() for waves in all_wavenumbers)
                                max_wave = min(waves.max() for waves in all_wavenumbers)
                                n_points = min([len(w) for w in all_wavenumbers])
                                common_wavenumbers = np.linspace(min_wave, max_wave, n_points)
                                # Interpolate all spectra to common grid
                                interpolated_spectra = [
                                    np.interp(common_wavenumbers, w, i)
                                    for w, i in zip(all_wavenumbers, all_intensities)
                                ]
                                self.spectra = np.array(interpolated_spectra)
                                self.wavenumbers = common_wavenumbers
                                print(f"✓ Interpolated and converted spectra to array: {self.spectra.shape}")
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
                       normalization: bool = True,
                       outlier_percentile: float = 100.0,
                       min_samples_per_class: int = 2):
        """
        Preprocess the extracted spectral data
        
        Parameters:
        -----------
        wavenumber_range : tuple, optional
            (min, max) wavenumber range to keep
        normalization : bool
            Whether to normalize spectra
        outlier_percentile : float
            Percentile threshold for outlier removal (100 disables)
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
            print(f"\u2713 Savitzky–Golay smoothing applied (win={win}, poly={order})")
        
        if use_derivative:
            # SG derivative keeps noise low and aligns grid spacing
            deriv = np.asarray([
                savgol_filter(row, 15, 3, deriv=1) for row in spectra
            ])
            spectra = np.hstack([spectra, deriv])      # concat orig + dI/dν
            if wavenumbers is not None:
                wavenumbers = np.hstack([wavenumbers, wavenumbers])  # dummy duplicate
            print("\u2713 First-derivative features appended")

        # ------------------------------------------------------------------
        # 3.5) OPTIONAL NORMALIZATION
        # ------------------------------------------------------------------
        if normalization:
            spectra = (spectra - np.mean(spectra, axis=1, keepdims=True)) / (np.std(spectra, axis=1, keepdims=True) + 1e-8)
            print("\u2713 Spectra normalized (z-score)")

        # ------------------------------------------------------------------
        # 4) VARIANCE FILTER – DROP FLATTEST 20 % OF CHANNELS
        # ------------------------------------------------------------------
        var = np.var(spectra, axis=0)
        mask_var = var > np.percentile(var, 40)
        spectra  = spectra[:, mask_var]
        if wavenumbers is not None:
            wavenumbers = wavenumbers[mask_var]
        print(f"\u2713 Variance filter kept {spectra.shape[1]} informative channels")

        # ------------------------------------------------------------------
        # 5) CLASS PRUNING (min_samples_per_class)
        # ------------------------------------------------------------------
        unique, counts = np.unique(labels, return_counts=True)
        keep_classes   = unique[counts >= min_samples_per_class]
        if len(keep_classes) < len(unique):
            keep_mask = np.isin(labels, keep_classes)
            spectra, labels = spectra[keep_mask], labels[keep_mask]
            print(f"\u2713 Removed classes with <{min_samples_per_class} samples")

        # ------------------------------------------------------------------
        # 6) OPTIONAL OUTLIER REMOVAL (percentile-based)
        # ------------------------------------------------------------------
        if outlier_percentile < 100.0:
            median_spec = np.median(spectra, axis=0)
            mad_vec     = np.median(np.abs(spectra - median_spec), axis=1)
            thr         = np.percentile(mad_vec, outlier_percentile)
            ok          = mad_vec < thr
            spectra, labels = spectra[ok], labels[ok]
            print(f"\u2713 Removed {np.sum(~ok)} spectral outliers (>{outlier_percentile} percentile)")

        # ------------------------------------------------------------------
        # 6.5) FINAL CLASS PRUNING (ensure all classes have >=2 samples)
        # ------------------------------------------------------------------
        unique, counts = np.unique(labels, return_counts=True)
        keep_classes   = unique[counts >= 2]
        if len(keep_classes) < len(unique):
            keep_mask = np.isin(labels, keep_classes)
            spectra, labels = spectra[keep_mask], labels[keep_mask]
            print(f"\u2713 Final removal of classes with <2 samples after all preprocessing")

        # ------------------------------------------------------------------
        # 7) SAVE BACK
        # ------------------------------------------------------------------
        self.processed_data = {
            'spectra'      : spectra,
            'labels'       : labels,
            'wavenumbers'  : wavenumbers,
            'mineral_names': np.unique(labels)
        }
        print("\n\u2713 Preprocessing complete!")
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
    
    def parse_enlighten_csv(csv_str: str) -> dict:
    """
    Parse ENLIGHTEN spectrometer CSV string data
    
    Parameters:
    -----------
    csv_str : str
        Raw CSV string from ENLIGHTEN spectrometer
        
    Returns:
    --------
    result : dict
        Dictionary containing:
        - 'metadata': dict of header parameters
        - 'wavenumbers': array of wavenumbers
        - 'intensities': array of processed intensities
    """
    lines = csv_str.split('\r\n')
    metadata = {}
    data_start = None
    
    # Parse metadata header
    for i, line in enumerate(lines):
        if line.startswith('Pixel,Wavelength,Wavenumber,Processed'):
            data_start = i + 1
            break
        if ',' in line:
            key, value = line.split(',', 1)
            metadata[key.strip()] = value.strip()
    
    # Parse spectral data
    wavenumbers = []
    intensities = []
    for line in lines[data_start:]:
        if not line or ',' not in line:
            continue
        parts = line.split(',')
        try:
            if len(parts) >= 4:
                wavenumbers.append(float(parts[2]))
                intensities.append(float(parts[3]))
        except ValueError:
            continue
    
    return {
        'metadata': metadata,
        'wavenumbers': np.array(wavenumbers),
        'intensities': np.array(intensities)
    }

def create_training_dataset(csv_data: dict, label: str) -> pd.DataFrame:
    """
    Create training dataset from parsed CSV data
    
    Parameters:
    -----------
    csv_data : dict
        Dictionary from parse_enlighten_csv
    label : str
        Mineral class label
        
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with wavenumbers as columns and single spectrum row
    """
    # Create DataFrame with wavenumber columns
    wavenumbers = csv_data['wavenumbers']
    intensities = csv_data['intensities']
    
    # Create column names
    col_names = [f"intensity_{w:.2f}" for w in wavenumbers]
    
    # Create single-row DataFrame
    df = pd.DataFrame([intensities], columns=col_names)
    df['label'] = label
    
    return df

def process_and_combine_csvs(csv_dict: Dict[str, str]) -> pd.DataFrame:
    """
    Process multiple CSV strings and combine into training dataset
    
    Parameters:
    -----------
    csv_dict : dict
        Dictionary of {filename: csv_str} pairs
        
    Returns:
    --------
    combined_df : pd.DataFrame
        Combined training dataset with spectra and labels
    """
    all_dfs = []
    
    for filename, csv_str in csv_dict.items():
        try:
            # Extract label from filename (filename_1, filename_2, etc.)
            label = filename.rsplit('_', 1)[0]
            
            # Parse CSV
            parsed = parse_enlighten_csv(csv_str)
            
            # Create DataFrame for this spectrum
            df = create_training_dataset(parsed, label)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    return pd.concat(all_dfs, ignore_index=True)
        