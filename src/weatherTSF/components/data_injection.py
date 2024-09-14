import os

import matplotlib as mpl

import pandas as pd
import tensorflow as tf
import zipfile


def getDataset():

    """
    Downloads and extracts a CSV dataset from a remote ZIP file, processes the data, and returns a DataFrame and a Series.

    The function performs the following steps:
    1. Downloads a ZIP file containing the dataset from a specified URL.
    2. Extracts the ZIP file into the current working directory.
    3. Reads the CSV file from the extracted contents into a DataFrame.
    4. Processes the DataFrame by slicing it to include every 6th row starting from the 5th row and converting the 'Date Time' column to datetime objects.
    5. Cleans up by removing the ZIP file and extracted CSV file.
    
    Returns:
        df (pd.DataFrame): Processed DataFrame containing the dataset with every 6th row.
        date_time (pd.Series): Series containing datetime objects converted from the 'Date Time' column.

    Raises:
        KeyError: If the 'Date Time' column is not found in the CSV file.
    """

    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=False)  


    extract_dir = os.path.dirname(zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        zip_ref.extractall(extract_dir)
        extracted_files = zip_ref.namelist()
    
    
    csv_files = [f for f in extracted_files if f.endswith('.csv')]
    
    
    if csv_files:
        csv_path = os.path.join(extract_dir, csv_files[0])
        df = pd.read_csv(csv_path)
        
        if 'Date Time' in df.columns:
            # sliceData
            df = df[5::6]
            date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
        else:
            raise KeyError("'Date Time' column not found in the CSV file.")
    
    
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    
    for file_name in extracted_files:
        file_path = os.path.join(extract_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)

    return df, date_time

def basicMplSetup():
    mpl.rcParams['figure.figsize'] = (8, 6)
    mpl.rcParams['axes.grid'] = False