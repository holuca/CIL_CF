# ALS
This code attempts to compute the missing values of a given sparse matrix with the Alternate Least Squares (ALS) method.
It uses the formula for the separable least squares solution (see formula 2.36 from the CIL lecture script 2024) to compute the updated values for the matrices U and V respectively.
The code written has been adapted from the presented snippets in the CIL script.
## Instructions for Use
1. Open the file `ALS.ipynb` in an editor that can handle jupyter notebooks.
2. Import the files `data_train.csv` and `sampleSubmission.csv` into the same directory as the notebook.
3. Change the filenames to import and export in the *Supporting Functions* section.
4. Choose desired parameters for the ALS algorithm in the *Import Matrix and Choose Parameters* section.
5. Run all code cells in sequence and receive the output file.