import pandas as pd
import numpy as np
import openpyxl
import re
import sys

DATABASE_FILENAME = "data/unmodified_database.xlsx"


def load_database():
    """Loads the unmodified_database excel file and cleans Medication column"""
    df = pd.read_excel(DATABASE_FILENAME)
    df["Medication"] = clean_series(df["Medication"])  # Retrieve only Medication names
    df = df[~df.Gene.str.contains(";")]  # Removes all rows with Gene containing ; e.g. CYP2C19;CYP2D6
    df.reset_index(drop=True, inplace=True)  # Resets index

    return df


def load_cpic():
    df = pd.read_excel("data/cpic_gene-drug_pairs.xlsx")
    df["Medication"] = pd.Series([medication.capitalize() for medication in df["Medication"]])
    return df


def clean_series(series):
    """Returns a series containing only a word"""
    only_word_regex = r"[\w]*"
    return pd.Series([re.match(only_word_regex, name)[0] for name in series])


def search(word, dataframe, remove_search_column=True):
    """Search given DataFrame"""
    series_output = pd.Series([False for i in range(len(dataframe))])
    found_in_key = []

    # print(dataframe.to_string())

    for key in dataframe.keys():
        series = dataframe[key].astype(str).str.match(word)
        if series.any():
            found_in_key.append(key)
            series_output = series_output + series

    output = dataframe[series_output]
    if remove_search_column:
        output = dataframe[series_output].loc[:, ~dataframe.columns.isin(found_in_key)]

    return output


def trunking(series_output, column):
    output_categorized = {}

    for i in range(len(series_output[column.capitalize()])):
        categories = list(series_output[column.capitalize()])[i].split(";")

        for category in categories:
            value_in_dict = output_categorized.get(category, [])
            value_in_dict.append(i)
            output_categorized[category] = value_in_dict

    series_output.drop([column.capitalize()], axis=1, inplace=True)

    return output_categorized


def printify(output_categorized, series_output):
    for key, value in output_categorized.items():
        pd.DataFrame(series_output.iloc[value].to_string(index=False))
        print(key)
        print(series_output.iloc[value].to_string(index=False))
        print()


if __name__ == '__main__':
    text = input("Enter value to search: ")
    search(text, load_database())
