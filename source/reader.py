import pandas as pd


def reader(path_to_file, sep='\t', headers=0):
    ''' reading and parsing of input data sample '''

    # method chaining
    df = (pd.read_csv(path_to_file, sep=sep, header=headers)
            .set_index('id_job')['features']
            .str.split(',', expand=True)
            .rename({x: f'feature_2_{x}' if x !=0 else 'feature_code'
                    for x in range(0, 257)}, axis=1)
            .applymap(lambda x: int(x))
            .reset_index()
            )
    return df


if __name__ == '__main__':
    path_to_file = 'data/train.tsv'
    df = reader(path_to_file=path_to_file)
    print(df.head())
