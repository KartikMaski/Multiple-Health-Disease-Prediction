import csv

def clean_and_convert_txt_to_csv(input_file, output_file):
    with open(input_file, 'r') as txt_file:
        lines = txt_file.readlines()

    # Open the output CSV file for writing
    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        for line in lines:
            # Remove leading/trailing quotes and split by commas
            cleaned_line = line.strip().strip('"')
            row = cleaned_line.split(',')

            # Write each cleaned row to the CSV file
            writer.writerow(row)

    print(f"Data has been successfully cleaned and saved to '{output_file}'")

# Specify input and output file paths
input_file = 'final.txt'  # Replace with your text file path
output_file = 'output_data.csv'

# Call the function
clean_and_convert_txt_to_csv(input_file, output_file)
