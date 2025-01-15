import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
import gdown

def preprocess_chunk(chunk):
    numeric_cols = ["Start_Lat", "Start_Lng", "End_Lat", "End_Lng", "Distance(mi)", 
                    "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)", 
                    "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]
    for col in numeric_cols:
        if col in chunk.columns:
            chunk[col] = chunk[col].fillna(chunk[col].mean())

    categorical_cols = ["Source", "Weather_Condition", "Sunrise_Sunset", "Civil_Twilight", 
                        "Nautical_Twilight", "Astronomical_Twilight"]
    for col in categorical_cols:
        if col in chunk.columns:
            chunk[col] = chunk[col].fillna('Unknown')

    if 'Start_Time' in chunk.columns:
        chunk['Start_Time'] = pd.to_datetime(chunk['Start_Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        chunk['Year'] = chunk['Start_Time'].dt.year


    columns_to_drop = ['ID', 'Start_Time', 'End_Time', 'Description', 'Number', 'Street', 
                       'Side', 'City', 'County', 'State', 'Zipcode', 'Country', 'Timezone', 
                       'Airport_Code', 'Weather_Timestamp']
    chunk.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    return chunk


def download_dataset(file_id, output_file):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    gdown.download(url, output_file, quiet=False)


def load_and_preprocess_data(file_path, chunk_size=10000):
    chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
    all_chunks = []
    for chunk in chunk_iter:
        preprocessed_chunk = preprocess_chunk(chunk)
        all_chunks.append(preprocessed_chunk)
    return pd.concat(all_chunks, ignore_index=True)

def main():
    file_id = "1rpHorzGYokgiVKv0CyodiqPP7UdrS6eZ"
    output_file = "data.csv"
    download_dataset(file_id, output_file)

    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(output_file)

    df.dropna(subset=['Severity'], inplace=True)
    df['Severity'] = df['Severity'].astype(int)

    X = df.drop('Severity', axis=1)
    y = df['Severity']

    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values for numerical columns
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values for categorical columns
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    print("Training the model...")
    model.fit(X_train, y_train)

    print("Evaluating the model...")
    y_pred = model.predict(X_test)       

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

if __name__ == "__main__":
    main()
