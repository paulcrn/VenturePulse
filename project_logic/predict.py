import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def train_model():
    df = pd.read_csv('raw_data/features_enriched_data.csv',
                     parse_dates=['founded_at', 'first_funding_at', 'last_funding_at'])

    df = df[df['status_enriched'] != 'unknown']
    df['target'] = (df['status_enriched'].isin(['acquired', 'operating'])).astype(int)

    engineered_num = [
        'avg_raised_per_round',
        'age_first_funding_days',
        'has_multiple_rounds',
        'funding_span_days',
        'avg_years_between_rounds',
    ]

    raw_num = [
        'funding_rounds',
        'funding_total_usd',
        'seed',
        'venture',
        'angel',
        'grant',
        'debt_financing',
        'private_equity',
        'round_A',
        'round_B',
        'round_C',
        'round_D',
        'round_E',
    ]

    cat_features = ['region_group', 'industry_group']

    X = df[engineered_num + raw_num + cat_features].copy()
    y = df['target'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ('num', RobustScaler(), engineered_num + raw_num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
    ])

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('model', LogisticRegression(
            C=0.1,
            penalty='l1',
            solver='liblinear',
            class_weight='balanced',
            max_iter=2000,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)
    print("Model trained successfully")
    return pipeline


def predict(pipeline, input_data: dict) -> float:
    X = pd.DataFrame([input_data])
    proba = pipeline.predict_proba(X)
    return float(proba[0][1])
