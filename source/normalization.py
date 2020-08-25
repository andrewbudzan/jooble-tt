import pandas as pd

def z_score(df):
    '''
    basic formula:
        (value - dataset mean) / sd
    in our case:
        value - numerical characteristics of vacancy
        dataset mean - each column mean value
        sd - standard deviation
    '''
    # colums from dataframe, which will be used for this operation
    use_cols = [f'feature_2_{x}' for x in range(1, 257)]

    # because of all features are numericalm we can call
    # DataFrame.describe() method and save all values we need
    df_stat = df[use_cols].describe().loc[['mean', 'std'], :]

    return df.reset_index()[use_cols].apply(lambda x:
                        (x - df_stat.loc['mean', :]) / df_stat.loc['mean', :],
                        axis=1)


if __name__ == '__main__':
    from reader import reader
    path_to_file = 'data/train.tsv'
    df = reader(path_to_file=path_to_file)
    print(z_score(df).head())
