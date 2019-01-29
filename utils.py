import numpy as np
import pandas as pd
import re
import datetime
import progressbar

from sklearn.externals import joblib
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


def impute(df):
    """
    Imputes the most frequent value per column.

    Args:
        df (dataframe) - dataframe to be used.
        columns (array) - array of column namnes.

    Returns:
        df_copy (dataframe) - imputed dataframe.
    """
    df_copy = df.copy()

    cnter = 0
    bar = progressbar.ProgressBar(maxval=df_copy.shape[1]+1, widgets=[
                                  progressbar.Bar('-', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    columns = df_copy.columns[df_copy.isnull().any()]

    for col in columns:
        cnter += 1
        bar.update(cnter)
        most_freq = df_copy.groupby([col])[col].count(
        ).sort_values(ascending=False).index[0]
        df_copy[col].fillna(most_freq, inplace=True)

    bar.finish()

    return df_copy

# ******************************************************************** #
# *********************** FEATURE ENGINEERING ************************ #
# ******************************************************************** #


def get_feature_types(df, feat_file):
    '''
    Divides features in df into different groups based on feature type.

    Args:
        df (dataframe) - dataframe to be used.
        feat_file (dataframe) - dataframe containing information about each feature.
    Returns:
        types (dict) - dictionary with type as key and list of features as value.
    '''
    types = dict()
    for col in df.columns:
        feat_type = feat_file[feat_file['attribute'] == col]['type'].values[0]
        if feat_type in types:
            types[feat_type].append(col)
        else:
            types[feat_type] = [col]

    return types


def categorical_info(df, features):
    '''
    Separates and stores categorical features in two dictionaries containing
    both the name of the column and the number of unique values.

    Args:
        df (dataframe) - dataframe in which the features are located.
        columns (array) - string array containing the features.

    Returns:
        binary (dict) - binary features.
        multiple (dict) - multi-level features.

    '''
    binary = {}
    multiple = {}
    for feat in features:
        values = df[feat].nunique()
        if values <= 2:
            binary[feat] = values
        else:
            multiple[feat] = values

    return binary, multiple



# function that converts original value into either decade or movement
def get_decade_movement(number):
    """
    Get corresponding two values inside a dictionary.

    Args:
        number (int): original number that will be used as a key to access the dictionary values.

    Returns:
        Two numbers (int) - corresponding numbers from dict.
    """
    decade_movement_dict = {
        '1': [1, 1],
        '2': [1, 0],
        '3': [2, 1],
        '4': [2, 0],
        '5': [3, 1],
        '6': [3, 0],
        '7': [3, 0],
        '8': [4, 1],
        '9': [4, 0],
        '10': [5, 1],
        '11': [5, 0],
        '12': [5, 1],
        '13': [5, 0],
        '14': [6, 1],
        '15': [6, 0]
    }
    number = int(number)
    return decade_movement_dict[str(number)][0], decade_movement_dict[str(number)][1]


def get_tens_ones_digits(number):
    """
    Convert XX number into it's 10th and 1th value.

    Args:
        number (int): original number.

    Returns:
        Two numbers (int): representing the original numbers 10th and 1th value.
    """
    return int(number/10), number // 10**0 % 10


def clean_data(df, feat_info, columns=None, customer_data=False, drop_rows=True):
    '''
    Perform feature trimming, re-encoding, and engineering on dataframe.

    Args: 
        df (dataframe) - dataframe to clean.
        feat_info (dataframe) - feature information
        columns (array) - array of columns to be present in dataframe after cleaning.
        customer_data (bool) - whether df is customer data or not.

    Returns: 
        df_cleaned (dataframe) - trimmed and cleaned dataframe.
    '''
    print('****** Step 1 - Load *****')
    print(f'Preparing to clean dataset with shape: {df.shape}\n')

    if customer_data:
        df.drop(['CUSTOMER_GROUP', 'ONLINE_PURCHASE', 'PRODUCT_GROUP'],
                axis=1,
                inplace=True)

    df.drop(['LNR', 'LP_LEBENSPHASE_FEIN', 'LP_LEBENSPHASE_GROB', 'CAMEO_DEU_2015'],
            axis=1,
            inplace=True)

    feat_info = feat_info[~feat_info['attribute'].isin(
        ['LNR', 'LP_LEBENSPHASE_FEIN', 'LP_LEBENSPHASE_GROB', 'CAMEO_DEU_2015'])]

    print(f'Dropped these features from dataframe and feat_info: ')
    print(['LNR', 'LP_LEBENSPHASE_FEIN', 'LP_LEBENSPHASE_GROB', 'CAMEO_DEU_2015'])
    print(f'Shape after Step 1 - Load: {df.shape}\n')

    print('****** Step 2 - Convert NaN codes *****')

    df_parsed = convert_missing_codes(df, feat_info)

    print(f'Shape after Step 2 - Convert NaN codes: {df_parsed.shape}\n')

    print('****** Step 3 - Drop Columns with Missing values >= 30 % *****')

    columns_drop = ['AGER_TYP',
                    'ALTER_HH',
                    'ALTER_KIND1',
                    'ALTER_KIND2',
                    'ALTER_KIND3',
                    'ALTER_KIND4',
                    'D19_BANKEN_ANZ_12',
                    'D19_BANKEN_ANZ_24',
                    'D19_BANKEN_DATUM',
                    'D19_BANKEN_DIREKT',
                    'D19_BANKEN_GROSS',
                    'D19_BANKEN_LOKAL',
                    'D19_BANKEN_OFFLINE_DATUM',
                    'D19_BANKEN_ONLINE_DATUM',
                    'D19_BANKEN_REST',
                    'D19_BEKLEIDUNG_GEH',
                    'D19_BEKLEIDUNG_REST',
                    'D19_BILDUNG',
                    'D19_BIO_OEKO',
                    'D19_DIGIT_SERV',
                    'D19_DROGERIEARTIKEL',
                    'D19_ENERGIE',
                    'D19_FREIZEIT',
                    'D19_GARTEN',
                    'D19_GESAMT_ANZ_12',
                    'D19_GESAMT_ANZ_24',
                    'D19_GESAMT_DATUM',
                    'D19_GESAMT_OFFLINE_DATUM',
                    'D19_GESAMT_ONLINE_DATUM',
                    'D19_HANDWERK',
                    'D19_HAUS_DEKO',
                    'D19_KINDERARTIKEL',
                    'D19_KOSMETIK',
                    'D19_LEBENSMITTEL',
                    'D19_LOTTO',
                    'D19_NAHRUNGSERGAENZUNG',
                    'D19_RATGEBER',
                    'D19_REISEN',
                    'D19_SAMMELARTIKEL',
                    'D19_SCHUHE',
                    'D19_SONSTIGE',
                    'D19_TECHNIK',
                    'D19_TELKO_ANZ_12',
                    'D19_TELKO_ANZ_24',
                    'D19_TELKO_DATUM',
                    'D19_TELKO_MOBILE',
                    'D19_TELKO_OFFLINE_DATUM',
                    'D19_TELKO_ONLINE_DATUM',
                    'D19_TELKO_REST',
                    'D19_TIERARTIKEL',
                    'D19_VERSAND_ANZ_12',
                    'D19_VERSAND_ANZ_24',
                    'D19_VERSAND_DATUM',
                    'D19_VERSAND_OFFLINE_DATUM',
                    'D19_VERSAND_ONLINE_DATUM',
                    'D19_VERSAND_REST',
                    'D19_VERSI_ANZ_12',
                    'D19_VERSI_ANZ_24',
                    'D19_VERSICHERUNGEN',
                    'D19_VOLLSORTIMENT',
                    'D19_WEIN_FEINKOST',
                    'EXTSEL992',
                    'GEBURTSJAHR',
                    'KBA05_BAUMAX',
                    'KK_KUNDENTYP',
                    'TITEL_KZ']

    df_parsed.drop(columns_drop,
                   axis=1,
                   inplace=True)

    print(
        f'Shape after Step 3 - Drop Columns with Missing values >= 30 %: {df_parsed.shape}\n')

    print('****** Step 4 - Drop Rows with Missing values >= 30 % *****')

    if drop_rows:
        df_parsed, _ = split_df(df_parsed, threshold=210)

    print(
        f'Shape after Step 4 - Drop Rows with Missing values >= 30 %: {df_parsed.shape}\n')

    print('****** Step 5 - Impute Missing Values *****')
    df_imputed = impute(df_parsed)
    if df_imputed.isnull().sum().any() == False:
        message = f'Shape after Step 5 - Impute Missing Values: {df_imputed.shape}\n'
    else:
        message = f'Step 5 - Missing Values still Found in Dataset'
    print(message)
    print(f'AM NULL: {df_imputed["EINGEFUEGT_AM"].isnull().sum()}')
    print('****** Step 6 - Re-encode Binary Categorical Features *****')

    bin_dict = {'W': 1, 'O': 0}
    df_imputed['OST_WEST_KZ'] = df_imputed['OST_WEST_KZ'].map(bin_dict)

    print(
        f'Shape after Step 6 - Re-encode Binary Categorical Features: {df_imputed.shape}\n')

    print('****** Step 7 - Re-encode Multi-Level Categorical Features *****')

    cat_features = ['CAMEO_DEUG_2015',
                    'CJT_GESAMTTYP',
                    'D19_KONSUMTYP',
                    'D19_LETZTER_KAUF_BRANCHE',
                    'FINANZTYP',
                    'GEBAEUDETYP',
                    'GFK_URLAUBERTYP',
                    'KBA05_MAXHERST',
                    'KBA05_MAXSEG',
                    'LP_FAMILIE_FEIN',
                    'LP_FAMILIE_GROB',
                    'LP_STATUS_FEIN',
                    'LP_STATUS_GROB',
                    'NATIONALITAET_KZ',
                    'SHOPPER_TYP',
                    'ZABEOTYP',
                    'WOHNLAGE']

    df_ohe = pd.get_dummies(df_imputed, columns=cat_features)

    print(
        f'Shape after Step 7 - Re-encode Multi-Level Categorical Features: {df_ohe.shape}\n')

    print('****** Step 8 - Re-encode Mixed Features *****')

    df_ohe["EINGEFUEGT_AM"] = pd.to_datetime(df_ohe["EINGEFUEGT_AM"], format='%Y/%m/%d %H:%M')
    df_ohe["EINGEFUEGT_AM"] = df_ohe["EINGEFUEGT_AM"].dt.year
    df_ohe['CAMEO_INTL_2015'] = pd.to_numeric(df_ohe['CAMEO_INTL_2015'])
    df_ohe['DECADE'], df_ohe['MOVEMENT'] = zip(
        *df_ohe['PRAEGENDE_JUGENDJAHRE'].map(get_decade_movement))
    df_ohe['WEALTH'], df_ohe['LIFE_STAGE'] = zip(
        *df_ohe['CAMEO_INTL_2015'].map(get_tens_ones_digits))

    df_ohe.drop(['CAMEO_INTL_2015', 'PRAEGENDE_JUGENDJAHRE'],
                axis=1,
                inplace=True)

    print(
        f'Shape after Step 8 - Re-encode Mixed Features: {df_ohe.shape}\n')

    if columns is not None:
        cols_diff = np.setdiff1d(columns, df_ohe.columns)
        for col in cols_diff:
            print('****** Step 8 - Add Missing Columns *****')

            df_ohe[col] = 0.0
            df_ohe[col] = df_ohe[col].astype('float')

            print(
                f'Shape after Step 8 - Add Missing Columns: {df_ohe.shape}\n')
    return df_ohe


# ******************************************************************** #
# **************************** PCA/KMEANS **************************** #
# ******************************************************************** #


def clusters_predict(df):
    '''
    Applies scaler, pca and predicts clusters on cleaned dataset.

    Args:
        df (dataframe) - dataframe to be used.

    Returns:
        df_clusters (dataframe) - dataframe containing LNR and predicted clusters.
    '''
    LNR = np.zeros(df.shape[0])
    if 'LNR' in df.columns:
        LNR = df['LNR'].values
        df.drop(['LNR'], axis=1, inplace=True)

    scaler = joblib.load('models/scaler3.save')
    pca = joblib.load('models/pca3.save')
    kmeans = joblib.load('models/kmeans3.save')

    df_scaled = scaler.transform(df.astype('float'))
    df_pca = pca.transform(df_scaled)
    df_preds = kmeans.predict(df_pca)

    df_clusters = pd.DataFrame({'LNR': LNR, 'Clusters': df_preds})

    return df_clusters


def interpret_pca(df, pca, component):
    '''
    Maps each weight to its corresponding feature name and sorts according to weight.

    Args:
        df (dataframe): dataframe on which pca has been used on.
        pca (pca): pca object.
        component (int): which principal compenent to return

    Returns:
        df_pca (dataframe): dataframe for specified component containing the explained variance
                            and all features and weights sorted according to weight.
    '''
    df_pca = pd.DataFrame(columns=list(df.columns))
    df_pca.loc[0] = pca.components_[component]
    dim_index = "Dimension: {}".format(component + 1)

    df_pca.index = [dim_index]
    df_pca = df_pca.loc[:, df_pca.max().sort_values(ascending=False).index]

    ratio = np.round(pca.explained_variance_ratio_[component], 4)
    df_pca['Explained Variance'] = ratio

    cols = list(df_pca.columns)
    cols = cols[-1:] + cols[:-1]
    df_pca = df_pca[cols]

    return df_pca

# ******************************************************************** #
# ***************************** PLOTTING ***************************** #
# ******************************************************************** #


def plot_difference_missing(df, columns, sort, threshold):
    '''
    Plots the amount of missing values before and after converting missing codes 
    values per column. 

    Args:
        df (dataframe) - dataframe containing columns.
        columns (list) - list of columns.
        sort (string) - column to sort by.
        threshold (int) - number of features to include.
    '''
    df.sort_values(sort, ascending=False)[columns][:threshold].plot(kind='bar', figsize=(20,10))


def plot_comparison(column, customer, general, clusters=[7,2]):
    '''
    Plots the distribution for a feature.
    
    Args:
        column (string) - feature to plot.
        customer (dataframe) - customer population cluster.
        general (dataframe) - general population cluster.
        clusters (list) - ints to use when pritting out title.
    '''   
    sns.set(style="darkgrid")
    fig, (ax1, ax2) = plt.subplots(figsize=(12,4), ncols=2)
    sns.countplot(x=column, data=customer, ax=ax1, palette="Set2")
    ax1.set_xlabel('Value')
    ax1.set_title(f'Distribution Customer population (cluster: {clusters[0]})')
    sns.countplot(x=column, data=general, ax=ax2, palette="Set2")
    ax2.set_xlabel('Value')
    ax2.set_title(f'Distribution General population (cluster: {clusters[1]})')
    fig.suptitle(f'Feature: {column}')


def plot_comparison2(column, df):
    '''
    Plots the distribution for a feature.

    Args:
        feature (string) - feature to plot.
        df (dataframe) - dataframe containing RESPONSE feature.
    '''
    responded = df[df['RESPONSE'] == 1]
    not_responded = df[df['RESPONSE'] == 0]

    sns.set(style="darkgrid")
    fig, (ax1, ax2) = plt.subplots(figsize=(12,4), ncols=2)
    sns.countplot(x=column, data=responded, ax=ax1, palette="Set2")
    ax1.set_xlabel('Value')
    ax1.set_title(f'Distribution for Responded = 1')
    sns.countplot(x=column, data=not_responded, ax=ax2, palette="Set2")
    ax2.set_xlabel('Value')
    ax2.set_title(f'Distribution for Responded = 0')
    fig.suptitle(f'Feature: {column}')


def plot_missing(df, column, threshold=75):
    '''
    Plots the amount of missing values per column. 

    Args:
        df (dataframe) - dataframe containing columns.
        column (string) - column to plot. Either before or after parsing missing values.
        threshold (int) - number of features to include.
    '''

    df_missing = df.sort_values(column, ascending=False)[:threshold]
    fig = plt.figure(figsize=(20,5))

    ax = sns.barplot(df_missing.index, df_missing[column], color = sns.color_palette()[2])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set(title=f'Missing values {column} converting codes to NaN', ylabel='# Missing values')


def plot_pca(dim, num):
    '''
    Plots the "top" and "bottom" 4 features and weights.

    Args:
        dim (row) - dimension from pca dataframe.
        num (int) - which dimension/compenent.
    '''
    features = dim.iloc[:, np.r_[1:5, -4, -3, -2, -1]]
    feature_names = features.columns
    weights = features.iloc[0].values

    sns.set(style='whitegrid')
    sns.set_color_codes('pastel')
    fig = plt.figure(figsize=(10, 5))
    sns.set()

    ax = sns.barplot(x=weights, y=feature_names)
    ax.set(xlabel="Weight", ylabel="Feature", title=f'Dimension {num}')


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


def plot_feature_importances(model, model_type, features, plot_n=10):
    '''
    Plots n most important features and importance.
    
    Args:
        model (classifier) - trained model.
        model_type (str) - type of model.
        features (list) - list of feature names.
        plot_n (int) - number of features to plot.
    '''
    feature_importance_values= np.zeros((len(model.feature_importances_)))
    
    feature_importance_values += model.feature_importances_

    feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

    # sort based on importance
    feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index(drop = True)

    # normalize the feature importances to add up to one
    feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
    feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])
    
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()
    
    ax.barh(list(reversed(list(feature_importances.index[:plot_n]))), 
                feature_importances['normalized_importance'][:plot_n], 
                align = 'center', edgecolor = 'k')

    # Set ticks and labels
    ax.set_yticks(list(reversed(list(feature_importances.index[:plot_n]))))
    ax.set_yticklabels(feature_importances['feature'][:plot_n], size = 12)
    plt.xlabel('Normalized Importance', size = 15); plt.title(f'Feature Importances ({model_type})', size = 15)