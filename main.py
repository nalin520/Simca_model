import argparse
from rruff.analysis import run_complete_rruff_simca_analysis
import json
import os

def read_csv_files(csv_paths):
    """Read multiple CSV files into strings"""
    csv_strings = []
    for path in csv_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
        with open(path, 'r') as f:
            csv_strings.append(f.read())
    return csv_strings

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run SIMCA analysis on CSV spectral data")
    
    # CSV input parameters
    ap.add_argument("--csv_files", required=True, nargs='+', 
                   help="List of CSV file paths containing spectral data")
    ap.add_argument("--start", type=float, default=0,
                   help="Start of wavenumber range (cm⁻¹)")
    ap.add_argument("--end", type=float, default=4000,
                   help="End of wavenumber range (cm⁻¹)")
    ap.add_argument("--num_wavenums", type=int, default=2000,
                   help="Number of wavenumber points for interpolation")
    
    # Class filtering
    ap.add_argument("--min_samples_per_class", type=int, default=10, 
                   help="Minimum samples per class to keep")
    ap.add_argument("--top_n_classes", type=int, default=10, 
                   help="Number of most common classes to use")
    
    # Feature block flags
    ap.add_argument("--no_deriv", action="store_true", 
                   help="Disable derivative features")
    ap.add_argument("--no_peak_stats", action="store_true", 
                   help="Disable peak statistics features")
    ap.add_argument("--no_region_ratios", action="store_true", 
                   help="Disable region ratio features")
    ap.add_argument("--no_pca", action="store_true", 
                   help="Disable PCA features")
    ap.add_argument("--no_entropy", action="store_true", 
                   help="Disable entropy features")
    ap.add_argument("--no_percentiles", action="store_true", 
                   help="Disable percentile features")
    ap.add_argument("--no_moments", action="store_true", 
                   help="Disable statistical moments features")
    
    # Preprocessing
    ap.add_argument("--normalization", type=str, default="area",
                   choices=["area", "max", "none"],
                   help="Normalization method")
    ap.add_argument("--outlier_percentile", type=float, default=10.0,
                   help="Percentile for outlier removal (0-100)")
    
    args = ap.parse_args()

    # Read CSV files
    csv_strings = read_csv_files(args.csv_files)
    
    # Extract filenames (mineral names will be derived from these)
    file_names = [os.path.basename(path) for path in args.csv_files]
    
    # Run analysis
    model, wavenumbers = run_complete_rruff_simca_analysis(
        csv_strings=csv_strings,
        file_names=file_names,
        start=args.start,
        end=args.end,
        num_wavenums=args.num_wavenums,
        min_samples_per_class=args.min_samples_per_class,
        top_n_classes=args.top_n_classes,
        include_deriv=not args.no_deriv,
        include_peak_stats=not args.no_peak_stats,
        include_region_ratios=not args.no_region_ratios,
        include_pca_features=not args.no_pca,
        include_entropy=not args.no_entropy,
        include_percentiles=not args.no_percentiles,
        include_moments=not args.no_moments,
        normalization=args.normalization,
        outlier_percentile=args.outlier_percentile
    )
    
    print("\nAnalysis completed successfully!")
    print(f"Model saved to: simca_model_*.joblib")
    print(f"Metadata saved to: simca_metadata_*.json")