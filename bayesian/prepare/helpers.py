import pandas as pd
import tqdm
import numpy as np
import geopandas as gpd

# Function: To encode categorical columns or features
def encode_categorical_columns(df, encoding_legend):
    """
    Encodes specified categorical columns in a DataFrame using a provided encoding legend.

    Args:
        df: The Pandas DataFrame to encode.
        encoding_legend: A dictionary where keys are column names and values are dictionaries
                         mapping original values to encoded values.

    Returns:
        The encoded DataFrame.
    """
    df = df.copy()  # Work on a copy to avoid side effects

    for col_name, mapping in encoding_legend.items():
        if col_name in df.columns:
            # Create new column name with '_encoded' suffix
            encoded_col_name = col_name + '_encoded'

            # Apply mapping and handle potential errors and assumes
            # that the missing values or "don't know" values were already
            # handled in data cleaning stage, so it should be safe to coerce them to pd.NA directly.
            df[encoded_col_name] = df[col_name].map(mapping).astype(pd.Int64Dtype())

    return df

# ============================================
# Decoding
# ============================================
def decode_dataframe(df, encoding_legend):
    """
    Decodes a DataFrame using an encoding legend.

    Args:
        df: The DataFrame to decode.
        encoding_legend: A dictionary mapping column names to encoding dictionaries.

    Returns:
        The decoded DataFrame.
    """
    df = df.copy()  # Create a copy to avoid in-place modification
    for col, mapping in encoding_legend.items():  # FIXED: removed .items from encoding_legend.items
        # Check if the column exists in the DataFrame before proceeding
        if col in df.columns:
            inverse_mapping = {v: k for k, v in mapping.items()}
            # Map the encoded values back to their original labels
            df[col] = df[col].map(inverse_mapping)

            # Handle any unmapped values (NaN) - optional logging
            unmapped = df[col].isna().sum()
            if unmapped > 0:
                print(f"Warning: {unmapped} unmapped values found in column '{col}'")

    return df

# Function: Identifying the unique values of a dataframe
def get_unique_values_with_nan(dataframe, column_name):
    """
    Prints the unique values of a specified column in a Pandas DataFrame,
    including NaN or blanks.

    Args:
        dataframe: The Pandas DataFrame.
        column_name: The name of the column to extract unique values from.
    """
    unique_values = dataframe[column_name].unique()
    print(f"Unique values in column '{column_name}':")

    # Print each unique value, handling NaN separately
    for value in unique_values:
        if pd.isnull(value):  # Check if the value is NaN
            print("NaN")    # Print NaN if it is
        else:
            print(value)     # Print the actual value otherwise

    print("-" * 20)  # Separator between columns

    # Function: Further Checks for the Dataframe
def check_missing_values_row(df):
    """
    Checks for missing values in a DataFrame and returns the indices of rows with missing values.
    Also prints the shape of the original DataFrame.

    Args:
        df: The Pandas DataFrame to check.

    Returns:
        list: List of indices of rows with missing values.
    """
    rows_with_missing_values = df[df.isnull().any(axis=1)].index
    print("Index of rows with missing values:", rows_with_missing_values.tolist())
    print("Original DataFrame shape:", df.shape)
    return rows_with_missing_values.tolist() # Return the list of indices

def check_missing_values_per_column(df):
    """
    Checks for missing values in each column of a DataFrame and prints the count of missing values per column.
    Also prints the total number of rows in the DataFrame.

    Args:
        df: The Pandas DataFrame to check.
    """
    for col in df.columns:
        print(col, df[col].isnull().sum())
    print("Total number of columns:", df.shape[1])

def check_column_data_types(df):
    """
    Checks and prints the data type of each column in a DataFrame.

    Args:
        df: The Pandas DataFrame to check.
    """
    for col in df.columns:
        print(col, df[col].dtype)

def check_numeric_columns(df):
    """
    Checks and prints whether each column in a DataFrame is numeric or not.

    Args:
        df: The Pandas DataFrame to check.
    """
    for column_name in df.columns:
        print(f'{column_name}: {pd.api.types.is_numeric_dtype(df[column_name])}')

def print_rows_with_missing_values(df):
    """
    Prints the complete rows where at least one missing value is present.

    Args:
        df: The Pandas DataFrame.
    """

    # Filter for rows with missing values
    rows_with_missing_values = df[df.isnull().any(axis=1)]

    # Print the selected rows (all columns)
    for index, row in rows_with_missing_values.iterrows():
        print(f"Row index: {index}")
        print(row.to_dict())  # Print as a dictionary for better formatting
        print("-" * 20)  # Separator between rows

# Converting all values to int type
def convert_column_to_int(dataframe, column_name):
    """
    Converts all values in the specified column of the DataFrame to integers.

    Args:
        dataframe: The Pandas DataFrame.
        column_name: The name of the column to convert.
    """
    try:
        # Using the to_numeric function to convert column to numeric type (int or float)
        dataframe[column_name] = pd.to_numeric(dataframe[column_name], errors='coerce')

        # Then convert the numeric data type to Integer
        dataframe[column_name] = dataframe[column_name].astype(int)

        print(f"Column '{column_name}' converted to integer type.")

    except ValueError:
        print(f"Error: Column '{column_name}' could not be converted to integer type. Some values may not be numeric.")