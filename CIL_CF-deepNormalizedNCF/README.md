# CIL Collaborative Filtering Project

Here are the attempts using a Deep Neaural Network with regularization methods including dropout-layers, adversarial noise and L1-regularization in order to combat overfitting. Bigger files (csv-files) like the formatted input files are mostly left out.

## Regularized Deep NN

(1. Format data into a csv file of the form: user_id, item_id, rating (e.g. 44,1,4) using `format_data2.py`)

2. Training and Testing are run from `main.py`. To select training or testing mode, use "train" or "test" as the second function input on the last line in main.py and above that have "input_csv = './data_train.csv'" commeted out for testing mode or "input_csv = './data_sampleSubmission.csv'" commented out for training mode

3. Training mode will output a .pth file for each epoch in the "checkpoints" directory.

4. To create a .csv file in submission format, move the .pth file corresponding to the epoch you want (presumably the last one) to the directory "best_checkpoint" and rename the file to "checkpoint.pth", then run testing mode. The submission file will be created under the name "results_current.csv".
