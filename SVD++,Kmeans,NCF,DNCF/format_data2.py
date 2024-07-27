import csv
import re



# Function to parse the custom format string
def parse_custom_format(custom_str):
    match = re.match(r'r(\d+)_c(\d+)', custom_str)
    if match:
        row = match.group(1)
        col = match.group(2)
        return row, col
    else:
        raise ValueError(f"Invalid format: {custom_str}")


def main(input_file, output_file):
    # Read the input CSV file and write the formatted data to the output CSV file
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
    
        # Skip the headers
        headers = next(reader, None)
    
        for row in reader:
            if row and len(row) > 1:  # Ensure the row is not empty and has both columns
                try:
                    id_part, prediction = row
                    row_num, col_num = parse_custom_format(id_part)
                    formatted_row = [row_num, col_num, prediction]
                    writer.writerow(formatted_row)
                except ValueError as e:
                    print(f"Skipping row due to parsing error: {row} - {e}")



if __name__ == "__main__":
    input_csv = './input_data/sampleSubmission.csv'
    output_csv = './data_sampleSubmission1.csv'


    
    main(input_csv, output_csv)
    print(f"Formatted data has been written to {output_csv}")