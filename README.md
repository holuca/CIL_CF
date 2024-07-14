# CIL Collaborative Filtering Project

Here are the attempts using SVD++ and Neural Collaborative Filtering. Bigger files like the formatted input files are left out.

1. Format data into a csv file of the form: user_id, item_id, rating (e.g. 44,1,4) using format_data2.py 
2. To run either svdpp.py or nerualCy import the specific file(line 11 in main, as forward has the same name sry for the ugliness), set the parameter in main to "train" or "test"
3. For "train", if using svd++ add parameter "dictionary" to the model, nerual CF can be used as it is now. IF you run multiple configurations at once, to save the best checkpoint uncomment in the trainable funciton accordingly that part. 
4. For "test", adjust the best_config file accordingly depending on what params you are testing, change training input file to the testing file. 
