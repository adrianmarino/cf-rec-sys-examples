def group_by(df, column):
    return df.groupby(column)['rating'].count().reset_index(name='count')

def most_common(df, column, limit):
    return df[column].value_counts()[:limit].index.tolist()
