import pandas as pd;
def preprocess_data(df):
    # Handle categorical features by encoding them
    df['season'] = df['season'].astype('category')
    df['yr'] = df['yr'].astype('category')  # Use 'yr' instead of 'year'
    df['mnth'] = df['mnth'].astype('category')  # Use 'mnth' instead of 'month'
    df['hr'] = df['hr'].astype('category')  # Use 'hr' instead of 'hour'
    df['weathersit'] = df['weathersit'].astype('category')  # Use 'weathersit' instead of 'weather'
    df['holiday'] = df['holiday'].astype('category')
    df['workingday'] = df['workingday'].astype('category')  # Use 'workingday' instead of 'working_day'

    # Check if the columns exist before dropping them
    if 'datetime' in df.columns:
        df = df.drop(['datetime'], axis=1)
    if 'atemp' in df.columns:
        df = df.drop(['atemp'], axis=1)

    # Split dataset into features (X) and target (y)
    X = df.drop('cnt', axis=1)  # Features
    y = df['cnt']  # Target (bike rental count)

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    return X, y
