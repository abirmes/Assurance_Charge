# Importing the pandas library
import pandas as pd
df = pd.read_csv('assurance_maladie.csv')
df.shape # Returns (number_of_rows, number_of_columns)
df.info() #serves as a quick and effective tool during EDA to get a
#high-level overview of the
#dataset's characteristics, data types, and the presence of missing values,
df.describe() #statistical summary of numerical columns, including count, mean, standard deviation, min, and max values.


duplicates = df.duplicated() # Get duplicates

df = df.drop_duplicates() #Remove Duplicates

missing_values = df.isnull().sum() #Identifying Missing Values: The isnull() method, combined with sum(), helps identify missing values in each column.




list = df.values.tolist()
