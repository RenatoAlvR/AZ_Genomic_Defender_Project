import pandas as pd
import numpy as np
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
import os
from pathlib import Path

def convert_csv_to_10x(input_csv_path, output_dir):
    """
    Converts a single CSV file (assumed: genes as rows, samples as columns, expression values)
    into 10x Genomics format (matrix.mtx, barcodes.tsv, features.tsv).

    Args:
        input_csv_path (Path): Path to the input CSV file.
        output_dir (Path): Directory where the output 10x files will be saved.
    """
    try:
        # --- 1. Read the CSV file ---
        # We assume the first column contains gene names, and subsequent columns
        # contain sample names with their corresponding expression values.
        print(f"Reading CSV: {input_csv_path.name}")
        df = pd.read_csv(input_csv_path, index_col=0)

        # Check if the DataFrame is empty after reading
        if df.empty:
            print(f"Warning: {input_csv_path.name} is empty. Skipping conversion.")
            return

        # --- 2. Extract features (gene names) ---
        # These are the row labels from the CSV
        features = df.index.tolist()
        features_path = output_dir / "features.tsv"
        with open(features_path, 'w') as f:
            for feature in features:
                f.write(f"{feature}\n")
        print(f"Saved features.tsv to {features_path}")

        # --- 3. Extract barcodes (sample names) ---
        # These are the column labels from the CSV
        barcodes = df.columns.tolist()
        barcodes_path = output_dir / "barcodes.tsv"
        with open(barcodes_path, 'w') as f:
            for barcode in barcodes:
                f.write(f"{barcode}\n")
        print(f"Saved barcodes.tsv to {barcodes_path}")

        # --- 4. Convert expression data to a sparse matrix ---
        # The 10x Genomics format uses a sparse matrix for efficiency,
        # storing only non-zero counts.
        # csr_matrix is a good format for this as it's efficient for matrix operations.
        # Ensure the data type is float32, which is commonly used for numerical precision.
        print("Converting data to sparse matrix...")
        sparse_matrix = csr_matrix(df.values, dtype=np.float32)

        # --- 5. Save the sparse matrix to Matrix Market format (.mtx) ---
        # The mmwrite function handles the specific format required for .mtx files,
        # including writing the header and 1-based indexing for coordinates.
        matrix_path = output_dir / "matrix.mtx"
        mmwrite(matrix_path, sparse_matrix)
        print(f"Saved matrix.mtx to {matrix_path}")

        print(f"Successfully converted {input_csv_path.name} to 10x Genomics format.")

    except Exception as e:
        print(f"Error converting {input_csv_path.name}: {e}")

def main():
    """
    Main function to iterate through CSV files in a user-specified input folder
    and convert each one into the 10x Genomics format.
    It creates a dedicated subfolder for each converted CSV's output files.
    """
    # Prompt the user for input and output folder paths
    input_folder_str = input("Enter the path to the folder containing your CSV files: ")
    output_base_folder_str = input("Enter the path to the base output folder where converted files will be saved: ")

    # Convert string paths to Path objects for easier manipulation
    input_folder = Path(input_folder_str)
    output_base_folder = Path(output_base_folder_str)

    # Validate the input folder
    if not input_folder.is_dir():
        print(f"Error: Input folder '{input_folder_str}' does not exist or is not a directory.")
        return

    # Create the base output folder if it doesn't exist.
    # `parents=True` creates any necessary parent directories, `exist_ok=True` prevents error if it already exists.
    output_base_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created/verified: {output_base_folder}")

    # Find all CSV files in the input folder
    csv_files = list(input_folder.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in '{input_folder_str}'. Please ensure your CSV files have a .csv extension.")
        return

    print(f"\nFound {len(csv_files)} CSV file(s) to convert.")

    # Iterate through each found CSV file and convert it
    for i, csv_file_path in enumerate(csv_files):
        print(f"\n--- Processing file {i+1}/{len(csv_files)}: {csv_file_path.name} ---")

        # Create a new subfolder within the base output folder for each CSV's output.
        # The subfolder name will be the same as the CSV file name (without extension).
        output_subfolder_name = csv_file_path.stem # `stem` gets the filename without its suffix
        current_output_dir = output_base_folder / output_subfolder_name
        current_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output subfolder: {current_output_dir}")

        # Call the conversion function for the current CSV
        convert_csv_to_10x(csv_file_path, current_output_dir)

    print("\nConversion process completed for all detected CSV files.")
    print(f"All converted 10x Genomics files are saved within subfolders inside: {output_base_folder}")

if __name__ == "__main__":
    main()
