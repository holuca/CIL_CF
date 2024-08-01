# CIL Collaborative Filtering Project

Here are the attempts using SVD++ and Neural Collaborative Filtering, Deeper Neural Collaborative Filtering and cosineSimilarity. Bigger files (csv-files) like the formatted input files are left out.

kMeans has been used for some trials of ensembling and seeing how it behaves but not on its own, which is actually just a cosine similarity of the sparse matrices for users or items. Ensembling is just merging all model results uniformly.

## SVD, Neural Collaborative Filtering, Deep Neural Collaborative Filtering

1. Format data into a csv file of the form: user_id, item_id, rating (e.g. 44,1,4) using `format_data2.py`
2. Training and Testing SVD++, NCF, DeepNCF are run from `main.py` by commenting out the specific import function for the models (as the train names are the same)
3. To run either `svdpp.py` or `nerualCF.py` import the specific file(line 11 in `main.py`, as forward has the same name sry for the ugliness), set the parameter in `main.py` to "train" or "test"
4. For "train", if using svd++ add parameter "dictionary" to the model, nerual CF can be used as it is now. IF you run multiple configurations at once, to save the best checkpoint uncomment in the trainable function accordingly that part. 
5. For "test", adjust the best_config file accordingly depending on what params you are testing, change training input file to the testing file.

If the main fucntions is called as it is without changing anything, it gets a prediction file (smth like results_neuralCFx.csv) using the current best checkpoint.pth in the best_checkpoint dir. and if only changed the "test" param in main.py to "train" and uncommenting the input_csv, it trains for the currently chosen config file the NeuralCF model.

## kMeans

This is a file which can be run on its own using the same formatted data as before. It outputs consine Similarity and has only been used for a new model of ensembling. The parameter can be changed depending of you want to choose user or item interaction. This code takes long as its just a for loop and not optmized, so if you try it just bear that in mind. The output needs to be formatted back using `format_back.py`

## Ensembling

`merge.py` takes all predictions of some chosen files which should be stored in one directory and averages the prediction columns out. The weights are uniform.
