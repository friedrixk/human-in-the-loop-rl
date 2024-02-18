import os
import pandas as pd

# Directory path
directory = "feedback"

# Get list of file names in the directory
file_names = os.listdir(directory)

# Initialize an empty list to store the DataFrames
dataframes = []

# Iterate over each file name
for file_name in file_names:
    # Create the file path
    file_path = os.path.join(directory, file_name)

    # Load the file as a DataFrame
    df = pd.read_csv(file_path)  # Assuming the files are CSV files, modify accordingly if different

    # Append the DataFrame to the list
    dataframes.append(df)




