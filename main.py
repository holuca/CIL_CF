import csv
import os

import numpy as np
import pandas as pd

# //----------------------------< read in dataset >----------------------------\\
# list declaration
r_vals = [] # row of entry
c_vals = [] # column of entry
rate_vals = [] # rating of entry

# read in training data
def readincsv():
    data = pd.read_csv('data_train.csv')
    datalist = data.values.tolist()
    for i in datalist:
        rc_val = i[0]
        r_text = rc_val.split('_')[0]
        c_text = rc_val.split('_')[1]
        r_num = (int) (r_text[1:])
        c_num = (int)(c_text[1:])
        r_vals.append(r_num)
        c_vals.append(c_num)

        rate_val = (int) (i[1])
        rate_vals.append(rate_val)
    return

readincsv()
# \\----------------------------< read in dataset >----------------------------//
