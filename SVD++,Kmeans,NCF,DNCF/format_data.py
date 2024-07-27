import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys

def convert_csv_to_matrix(input_csv, format):
    df = pd.read_csv(input_csv)
    
    df['row'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
    df['col'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
    
    max_row = df['row'].max()
    max_col = df['col'].max()

    if(format == "zero"):
        matrix = np.zeros((max_row, max_col))
        for index, row in df.iterrows():
            matrix[row['row']-1, row['col']-1] = row['Prediction']

    else:
        # Initialize and populate dictionary to store rows
        row_dict = {i: {} for i in range(1, max_row + 1)}
        for index, row in df.iterrows():
            row_dict[row['row']][row['col']] = row['Prediction']


        matrix = np.full((max_row, max_col), np.nan)
        for r in range(1, max_row + 1):
            for c in range(1, max_col + 1):
                if c in row_dict[r]:
                    matrix[r-1, c-1] = row_dict[r][c]
    
    return matrix

def mean_matrix(matrix):
    for r in range(matrix.shape[0]):
        row_mean = np.nanmean(matrix[r])
        matrix[r] = np.where(np.isnan(matrix[r]), row_mean, matrix[r])
    return matrix


def normalize_matrix(matrix):
    # Normalize the matrix by subtracting the row mean
    for r in range(matrix.shape[0]):
        row_mean = np.nanmean(matrix[r])
        matrix[r] = np.where(np.isnan(matrix[r]), row_mean, matrix[r]) - row_mean
    
    return matrix

def scale_matrix(matrix):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_matrix = scaler.fit_transform(matrix)
    
    return scaled_matrix

def save_matrix_to_csv(matrix, output_csv):
    matrix_df = pd.DataFrame(matrix)
    matrix_df.to_csv(output_csv, index=False, header=False)


def parsef(line):
    """ parses line and returns parsed row, column and value """
    l1 = line.decode('utf-8').split(',')
    l2 = l1[0].split('_')
    row = int(l2[0][1:])
    column = int(l2[1][1:])
    value = float(l1[1])
    return row, column, value


def loadRawData(file='../input_data/data_train.csv'):
    """ Loads and returns data in surprise format """
    itemID = []
    userID = []
    rating = []

    # parse data file into three arrays
    with open(file, 'rb') as f:
        content = f.readlines()
        content = content[1:]
        for line in content:
            if line:
                row, column, value = parsef(line)
                itemID.append(column)
                userID.append(row)
                rating.append(value)
    return itemID, userID, rating


def main(input_csv, output_csv, format):
    matrix = convert_csv_to_matrix(input_csv, format)
    
    if format == 'zero':
        pass
    elif format == 'mean':
        matrix = mean_matrix(matrix)
    elif format == 'normalize':
        # Normalize the matrix
        matrix = normalize_matrix(matrix)
    elif format == 'scale':
        # Normalize and scale the matrix
        matrix = normalize_matrix(matrix)
        matrix = scale_matrix(matrix)
    
    else:
        print(f"Unknown action: {format}")
        sys.exit(1)
    
    # Save the matrix to the output CSV
    save_matrix_to_csv(matrix, output_csv)



if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("<outputfile> should be string (.csv if needed), <format> should be 'zero', 'mean', 'normalize' or 'scale'")
        sys.exit(1)
    input_csv = './input_data/data_train.csv'
    output_csv = sys.argv[1]
    format = sys.argv[2]
    
    main(input_csv, output_csv, format)