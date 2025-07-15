import argparse
from rruff.analysis import run_complete_rruff_simca_analysis

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", required=True)
    ap.add_argument("--spectra_key", type=str, default=None, help="Key for spectra data (dict)")
    ap.add_argument("--labels_key", type=str, default=None, help="Key for mineral labels (dict)")
    ap.add_argument("--mineral_column", type=str, default=None, help="Column for mineral labels (DataFrame)")
    ap.add_argument("--spectrum_columns", type=str, nargs='+', default=None, help="Column(s) or pattern for spectra (DataFrame)")
    ap.add_argument("--min_samples_per_class", type=int, default=10, help="Minimum samples per class to keep")
    ap.add_argument("--top_n_classes", type=int, default=10, help="Number of most common classes to use")
    # Feature block flags
    ap.add_argument("--no_deriv", action="store_true", help="Disable derivative features")
    ap.add_argument("--no_peak_stats", action="store_true", help="Disable peak statistics features")
    ap.add_argument("--no_region_ratios", action="store_true", help="Disable region ratio features")
    ap.add_argument("--no_pca", action="store_true", help="Disable PCA features")
    ap.add_argument("--no_entropy", action="store_true", help="Disable entropy features")
    ap.add_argument("--no_percentiles", action="store_true", help="Disable percentile features")
    ap.add_argument("--no_moments", action="store_true", help="Disable statistical moments features")
    # Preprocessing
    ap.add_argument("--normalization", type=str, default="area", help="Normalization method: area, zscore, minmax")
    ap.add_argument("--outlier_percentile", type=float, default=10.0, help="Percentile for outlier removal (0-100)")
    args = ap.parse_args()

    run_complete_rruff_simca_analysis(
        args.pickle,
        spectra_key=args.spectra_key,
        labels_key=args.labels_key,
        mineral_column=args.mineral_column,
        spectrum_columns=args.spectrum_columns,
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
