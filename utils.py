import numpy as np
import pandas as pd
import datetime
import progressbar

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans

import matplotlib.pyplot as plt
import seaborn as sns


# ******************************************************************** #
# ************************** MISSING VALUES ************************** #
# ******************************************************************** #

def valid_values(df, missing_dict):
    '''
    Goes through the dataframe column by column and checks which values that are present.
    This information will be used when converting missing codes to NaN.

    ARGS:
        df (dataframe) - dataframe containing valid values.
        missing_dict (dict) - dictionary containing not valid values.

    RETURNS:
        valid_dict (dict) - dictionary with column as key and dict with valid values as value.
    '''
    valid_dict = dict()
    # skip LNR column
    for col in df.columns[1:]:
        val_dict = dict()
        for val in df[col].value_counts().index:
            if val not in val_dict and val not in missing_dict[col]:
                val_dict[val] = val
        valid_dict[col] = val_dict

    return valid_dict


def create_missing_dict(feat_file):
    '''
    Creates a dictionary with the feature name as key and it's
    corresponding missing value codes as value.

    ARGS:
        feat_file (dataframe) - dataframe containing information about feature.

    RETURNS:
        missing_dict (dict) - dictionary containing feature as name and missing as value
    '''
    missing_dict = {}
    for row in feat_file.itertuples():
        missing_dict[row.attribute] = eval(row.missing_or_unknown)

    return missing_dict


def convert_missing_codes(df, feat_file):
    '''
    Goes through the dataframe column by column and converts all the values
    in the feat_file to NaN.

    ARGS:
        df (dataframe) - dataframe which contains values to convert to NaN.
        feat_file (dataframe) - dataframe with information about each feature.

    RETURNS:
        df_copy (dataframe) - dataframe with values corresponding to missing value codes converted to NaN.
    '''

    missing_dict = create_missing_dict(feat_file)
    values = valid_values(df, missing_dict)

    cnter = 0
    bar = progressbar.ProgressBar(
        maxval=df.shape[1]+1, widgets=[progressbar.Bar('-', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    df_copy = df.copy()

    # skip LNR column
    for col in df.columns[1:]:
        cnter += 1
        bar.update(cnter)

        df_copy[col] = df_copy[col].map(values[col])

    bar.finish()

    return df_copy


def split_df(df, threshold):
    '''
    Splits dataframe into two new dataframes at a certain number of missing values. One df will contain all the rows
    that have at least the specified number of non missing values per row. The other df will contain the remaining rows.

    Args:
        df (dataframe): dataframe to split
        threshold (int): threshold to be used as splitting point.

    Returns: 
        two dataframes one containing (df_kept) all the rows that have at least the specified number of non
        NaN values and another (df_dropped) one containing the remaining rows.
    '''
    df_new = df.copy()
    df_kept = df_new.dropna(thresh=threshold)
    df_dropped = df_new[~df_new.index.isin(df_kept.index)]
    return df_kept, df_dropped


# ******************************************************************** #
# ***************************** PLOTTING ***************************** #
# ******************************************************************** #


def compare_columns(dfs, column):
    '''
    Plots the distribution of specified column for multiple dataframes to see if there is any difference.

    Args:
        dfs (array): an array that contains the dataframes to compare.
        column (str): column/feature which distribution will be plotted.

    Returns: None (plots)
    '''
    sns.set(style='whitegrid')
    sns.set_color_codes('pastel')
    fig = plt.figure(figsize=(20, 5))
    for i in range(1, 3):
        ax = fig.add_subplot(1, 2, i)
        g = sns.countplot(x=column, data=dfs[i-1])
        df_name = 'low missing' if i < 2 else 'high missing'
        plt.xlabel(column + ' ' + df_name)
        total = float(len(dfs[i-1]))
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 100,
                    '{:1.2f} %'.format(height/total * 100),
                    ha="center")
