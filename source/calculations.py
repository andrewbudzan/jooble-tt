import pandas as pd


def max_feature_index(df):
    ''' index of max feature value  '''

    # colums from dataframe, which will be used for this operation
    use_cols = [f'feature_2_{x}' for x in range(1, 257)]
    return df.reset_index()[use_cols].apply(lambda x: x.argmax(), axis=1)


def max_feature_abs_mean_diff(df):
    ''' absolute deviation of max value from mean '''

    # colums from dataframe, which will be used for this operation
    use_cols = [f'feature_2_{x}' for x in range(1, 257)]
    max_values_idx = max_feature_index(df[use_cols])
    max_values = pd.Series([df[use_cols].iloc[row, col]
                    for row, col in enumerate(max_values_idx)])
    df_max_vals = pd.Series([df[use_cols].iloc[x, y]
                    for x, y in enumerate(max_values_idx)])
    df_stat = df[use_cols].describe().loc[['mean'], :]
    df_mean_by_feature = pd.Series([df_stat.iloc[0, x]
                    for x in list(max_values_idx)])
    df_abs_mean = df_max_vals - df_mean_by_feature
    return df_abs_mean

if __name__ == '__main__':
    from reader import reader
    path_to_file = 'data/train.tsv'
    df = reader(path_to_file=path_to_file)
    print(max_feature_abs_mean_diff(df))
