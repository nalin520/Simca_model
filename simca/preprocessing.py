
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import os
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
import uuid
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, eye, diags


def savgol_smooth(y, window=15, polyorder=3):
    """Savitzky-Golay smoothing with safety checks."""
    window = min(len(y)//2*2+1, window)          
    if window < polyorder + 2:
        return y
    return savgol_filter(y, window, polyorder)

def median_MAD_scale(y, eps=1e-9):
    """Robust per-spectrum scaling: (y - median) / MAD."""
    med = np.median(y)
    mad = np.median(np.abs(y - med)) + eps
    return (y - med) / mad


def decode_payload_to_file(string_data: str, file_name: str):
    # Extract the file content from the payload
    file_content = string_data
    if file_name.split('.')[-1] == "txt":
        file_content = string_data.replace('\t', ',').replace('\s', ',')

    # Get the current working directory
    temp_folder = "app/ML/chat_bot/temp"
    # Create a file path that includes the current working directory
    file_path = os.path.join(temp_folder, f"{file_name.split('.')[0]}.csv")

    with open(file_path, 'w', newline='') as file:
        file.write(file_content)

    print(f"File '{file_path}' created successfully.")
    return file_path

def starting_index_and_check(lst):
    n = len(lst)
    # Consider adjusting or removing this length check if not needed
    if n < 200:
        return -1
    for i in range(0, n - 1):
        # Allow non-strict monotonicity
        if n - i >= 200:
            # Non-decreasing check
            if all(lst[j] <= lst[j + 1] for j in range(i, n - 1)):
                return i
            # Non-increasing check
            if all(lst[j] >= lst[j + 1] for j in range(i, n - 1)):
                return i
        else:
            break
    return -1

def difference_check(data):
    if not data.size:
        return False
    index = starting_index_and_check(data)
    if index == -1:
        return -1
    data = data[index:]
    differences = [abs(data[i + 1] - data[i]) for i in range(len(data) - 1)]
    max_diff = max(differences)
    min_diff = min(differences)
    difference = max_diff - min_diff
    # Adjust the difference threshold if necessary
    if 0 <= difference <= 100:
        return index
    return -1

def isWaveNumber(row):
    # Get the indices where NaN is False in the original row
    initial_non_nan_indices = np.where(~pd.isna(row))[0]

    # Extract the wavenumber values (ignoring NaN values) - will remove any non-numeric
    wavenumber_values = pd.to_numeric(row[initial_non_nan_indices], errors='coerce')

    # Now get the indices where wavenumber_values are not NaN (i.e., valid numeric values)
    valid_numeric_mask = ~pd.isna(wavenumber_values)
    indices = initial_non_nan_indices[valid_numeric_mask]  # Get the original indices of valid values
    # Filter out NaNs from wavenumber_values based on the same valid_numeric_mask
    wavenumber_values = wavenumber_values[valid_numeric_mask]

    # Now, valid_indices contains the original row indices of valid, numeric wavenumber_values

    # print(indices)
    index = difference_check(wavenumber_values)
    if index!=-1:
        return indices[index], wavenumber_values[index:]
    return -1, None

def parse_data(df):
    # Drop fully empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    # Apply string stripping only on string columns
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df.reset_index(drop=True, inplace=True)
    wavenumber_intensity_pairs = []
    i = 0
    while i < len(df):
        row = df.iloc[i].values
        # Check if the row contains Wavenumber data (ignore header and other irrelevant rows)
        if 'Pixels' not in row:
            start_index, wavenumber_values = isWaveNumber(row)
            if start_index != -1 and wavenumber_values.size > 0:
                # Extract the corresponding intensity values from subsequent rows
                j = i + 1
                last_intensity = None
                while j < len(df):
                    next_row = df.iloc[j].values
                    if isWaveNumber(next_row)[0] == -1:
                        # Adjust for potential length mismatches
                        intensity_values = next_row[start_index:]
                        min_length = min(len(wavenumber_values), len(intensity_values))
                        wavenumber_values = wavenumber_values[:min_length]
                        intensity_values = intensity_values[:min_length]
                        last_intensity = intensity_values
                        j += 1
                    else:
                        i = j - 1
                        break
                if last_intensity is not None:
                    wavenumber_intensity_pairs.append((wavenumber_values, last_intensity))
        i += 1
    if not wavenumber_intensity_pairs:
        raise ValueError("No Wavenumber-Intensity pair Found")
    # Generate a list of DataFrames for each wavenumber-intensity pair
    dataframes = []
    for wavenumber_values, intensity_values in wavenumber_intensity_pairs:
        parsed_df = pd.DataFrame({
            'Intensity': intensity_values
        }, index=wavenumber_values)
        # Filter Wavenumbers in the range 0 to 4000
        parsed_df = parsed_df[(parsed_df.index >= 0)]
        # Convert Wavenumber and Intensity to integers
        parsed_df.index = parsed_df.index.astype(int)
        parsed_df['Intensity'] = parsed_df['Intensity'].astype(int)
        # Rename the index to Wavenumber
        parsed_df.index.name = 'Wavenumber'
        dataframes.append(parsed_df)
    return dataframes

def parse_new_pca_mcr(string_data: str, file_name: str):
    file_path = decode_payload_to_file(string_data, file_name)
    df = pd.read_csv(file_path, header=None, delimiter=';', engine='python')

    # Split rows into separate columns
    split_data = df[0].str.split(',', expand=True)
    split_data = split_data.replace('', np.nan)

    def convert_to_numeric(col):
        try:
            return pd.to_numeric(col, errors='coerce')
        except Exception:
            return col

    try:
        # Convert columns to numeric where possible
        df_new = split_data.apply(convert_to_numeric)
        dataframes = parse_data(df_new)
        
    except Exception as e:
        print(f"Error in parsing data: {e}. Attempting transposed data parsing.")
        split_data = split_data.T
        df_new = split_data.apply(convert_to_numeric)
        try:
            dataframes = parse_data(df_new)
        except ValueError:
            print(f"Failed to find Wavenumber-Intensity pairs after transposing for file: {file_name}.")
            dataframes = []

    # Add suffix to filenames
    labeled_dataframes = [(f"{file_name.split('.')[0]}_{i + 1}.csv", df) for i, df in enumerate(dataframes)]

    # Ensure 'Wavenumber' is a column if it's an index in each DataFrame
    for i, (label, df) in enumerate(labeled_dataframes):
        if df.index.name == 'Wavenumber':
            df.reset_index(inplace=True)
            labeled_dataframes[i] = (label, df)

    return labeled_dataframes


def whittaker_smooth(x, w, lambda_, differences=1):
    """Smooth data using Whittaker smoothing."""
    X=np.matrix(x)
    m=X.size
    E=eye(m,format='csc')
    for i in range(differences):
        E=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*E.T*E))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)


def airPLS(x, dssn_th = 0.00001,lambda_=100, porder=1, itermax=10):
    """Perform adaptive iteratively reweighted penalized least squares for baseline fitting."""
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=whittaker_smooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn< dssn_th*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARNING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn)
        w[-1]=w[0]
    return z

def Baseline_Removal(combined_spectra):
    """Apply baseline removal using airPLS method and return original, baseline, and corrected spectra."""
    baseline_corrected_combined_spectra = []
    
    for spectra in combined_spectra:
        baseline = airPLS(spectra)
        BSremoval_specta = spectra - baseline
        baseline_corrected_combined_spectra.append((spectra, BSremoval_specta, baseline))  # Save original, corrected, and baseline
        
    return baseline_corrected_combined_spectra

def load_and_preprocess_data_from_strings(csv_strings, file_names, start, end, num_wavenums):
    """Load and preprocess data from multiple CSV strings using parse_new_pca_mcr."""
    all_spectra = []
    labels = []
    common_wavenum = np.linspace(start, end, num_wavenums)

    for string_data, file_name in zip(csv_strings, file_names):
        parsed_data = parse_new_pca_mcr(string_data, file_name)
        if not parsed_data:
            print(f"File {file_name} could not be parsed or contains no valid spectra. Skipping.")
            continue

        for suffix_filename, df in parsed_data:
            print(f"Processing {suffix_filename}...")

            if df.empty:
                print(f"{suffix_filename} is empty. Skipping.")
                continue

            if 'Wavenumber' not in df.columns or 'Intensity' not in df.columns:
                print(f"{suffix_filename} is missing required columns. Skipping.")
                continue

            df = df[(df['Wavenumber'] >= 0) & (df['Wavenumber'] <= 4000)]
            if df.empty:
                print(f"{suffix_filename} has no valid data in the range 0-4000. Skipping.")
                continue

            wavenum, intensity = df['Wavenumber'].values, df['Intensity'].values
            if len(wavenum) != len(intensity):
                print(f"Error: Wavenumber and intensity lengths do not match in {suffix_filename}. Skipping.")
                continue

            spectra = np.interp(common_wavenum, wavenum, intensity)
            all_spectra.append(spectra)
            labels.append(suffix_filename)

    if not all_spectra:
        print("No valid spectra found in any file.")
        return None, None, None

    combined_spectra = np.vstack(all_spectra)
    print(f"Processed {len(all_spectra)} valid spectra for further analysis.")

    baseline_corrected_combined_spectra = Baseline_Removal(combined_spectra)
    return common_wavenum, np.array(baseline_corrected_combined_spectra), labels