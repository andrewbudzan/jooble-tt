from source.reader import reader
from source.normalization import z_score
from source.calculations import max_feature_index, max_feature_abs_mean_diff

path_to_file = 'data/train.tsv'


def main(path, export=False):
    df_original = reader(path)
    df = z_score(df_original)
    df['max_feature_2_index'] = max_feature_index(df_original)
    df['max_feature_2_abs_mean_diff'] = max_feature_abs_mean_diff(df_original)
    df = df_original.loc[:, ['id_job']].join(df)
    if export:
        df.to_csv(str(export), sep='\t', index=False)
    else:
        print(f'from else: {export}')
    return df


if __name__ == '__main__':
    path_to_file = 'data/test.tsv'
    main(path_to_file, export='test_proc.tsv')
