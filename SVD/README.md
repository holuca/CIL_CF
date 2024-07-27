# ALS
This code attempts to compute the missing values of a given sparse matrix with different iterative SVD methods.
You can choose the algorithm to use, by changing the `mode` parameter. The three different options to choose from are:
* **singular_values_fixed**: SVP algorithm as described in equation (2.39) in the CIL script, A0 is chosen as the mean imputed input matrix. The desired rank can be chosen with `num_keep_singular_values`.
* **singular_values_shrink**: Modified SVP algorithm as described in equation (2.39) in the CIL script, A0 is chosen as the mean imputed input matrix. The rank of the matrix is determined by the shrink function and can be influenced by `shrinkage_tau`.
* **lecture_script_shrink**: Follows equation (2.41) in the CIL script, A0 is chosen as the mean imputed input matrix. The parameter `shrinkage_tau` can be chosen.
## Instructions for Use
1. Open the file `SVD.ipynb` in an editor that can handle jupyter notebooks.
2. Import the files `data_train.csv` and `sampleSubmission.csv` into the same directory as the notebook.
3. Change the filenames to import and export in the *Supporting Functions* section.
4. Choose desired parameters for the iterative SVD algorithms in the *Choose Parameters* section.
5. Run all code cells in sequence and receive the output file.