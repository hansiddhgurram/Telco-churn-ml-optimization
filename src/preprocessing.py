import pandas as pd

def load_and_clean_data(path: str) -> pd.DataFrame:
    """
    Load dataset and perform basic cleaning
    """
    df = pd.read_csv(path)

    # Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features
    """
    df = df.copy()

    # Avg monthly spend
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)

    # Tenure groups
    df['TenureGroup'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr']
    )

    # High monthly charge flag
    df['HighMonthlyCharge'] = (
        df['MonthlyCharges'] > df['MonthlyCharges'].median()
    ).astype(int)

    # Service count
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]

    df['ServiceCount'] = df[service_cols].apply(
        lambda row: sum(row == 'Yes'), axis=1
    )

    return df

def split_features_target(df: pd.DataFrame):
    """
    Split dataframe into X and y
    """
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    return X, y