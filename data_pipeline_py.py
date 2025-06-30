import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load data
def load_data(filepath):
    return pd.read_csv(filepath)

# Define preprocessing steps
def build_preprocessing_pipeline(df):
    # Identify numerical and categorical columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # Create transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers into a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor

# Main function to run pipeline
def preprocess_and_split_data(filepath, target_column):
    # Load data
    df = load_data(filepath)

    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Build the preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit and transform the training data, transform the test data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test

# Example usage
if __name__ == '__main__':
    filepath = 'your_data.csv'  # Replace with your actual CSV file path
    target_column = 'target'    # Replace with your actual target column name

    X_train, X_test, y_train, y_test = preprocess_and_split_data(filepath, target_column)

    print("Processed training data shape:", X_train.shape)
    print("Processed test data shape:", X_test.shape)
