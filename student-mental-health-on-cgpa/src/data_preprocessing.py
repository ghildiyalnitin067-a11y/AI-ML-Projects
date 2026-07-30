import os
import pandas as pd
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
    """Load the raw student mental health dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, standardize, and encode the student mental health dataset."""
    df = df.copy()

    # 1. Rename columns to standardized pythonic names
    column_mapping = {
        'Timestamp': 'timestamp',
        'Choose your gender': 'gender',
        'Age': 'age',
        'What is your course?': 'course',
        'Your current year of Study': 'year_of_study',
        'What is your CGPA?': 'cgpa',
        'Marital status': 'marital_status',
        'Do you have Depression?': 'depression',
        'Do you have Anxiety?': 'anxiety',
        'Do you have Panic attack?': 'panic_attack',
        'Did you seek any specialist for a treatment?': 'specialist_treatment'
    }
    df.rename(columns=column_mapping, inplace=True)

    # Drop timestamp if present as it's not predictive
    if 'timestamp' in df.columns:
        df.drop(columns=['timestamp'], inplace=True)

    # 2. Clean and impute 'age'
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    median_age = df['age'].median()
    df['age'] = df['age'].fillna(median_age).round().astype(int)

    # 3. Clean and standardize 'course'
    df['course'] = df['course'].astype(str).str.strip().str.lower()
    course_replacements = {
        'engine': 'engineering',
        'engin': 'engineering',
        'koe': 'engineering',
        'enm': 'engineering',
        'mathemathics': 'mathematics',
        'law': 'laws',
        'pendidikan islam': 'islamic education',
        'kirkhs': 'irkhs',
    }
    df['course'] = df['course'].replace(course_replacements)

    # 4. Clean 'year_of_study' -> Extract number
    df['year_of_study_num'] = (
        df['year_of_study']
        .astype(str)
        .str.extract(r'(\d+)')
        .fillna(1)
        .astype(int)
    )

    # 5. Clean and encode 'cgpa'
    df['cgpa'] = df['cgpa'].astype(str).str.strip()
    cgpa_ordinal_map = {
        '0 - 1.99': 1,
        '2.00 - 2.49': 2,
        '2.50 - 2.99': 3,
        '3.00 - 3.49': 4,
        '3.50 - 4.00': 5
    }
    cgpa_midpoint_map = {
        '0 - 1.99': 1.00,
        '2.00 - 2.49': 2.25,
        '2.50 - 2.99': 2.75,
        '3.00 - 3.49': 3.25,
        '3.50 - 4.00': 3.75
    }

    df['cgpa_ordinal'] = df['cgpa'].map(cgpa_ordinal_map).fillna(3).astype(int)
    df['cgpa_numeric'] = df['cgpa'].map(cgpa_midpoint_map).fillna(2.75).astype(float)

    # 6. Encode binary Yes/No variables
    binary_cols = ['marital_status', 'depression', 'anxiety', 'panic_attack', 'specialist_treatment']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # Gender binary encoding: Female = 0, Male = 1
    df['gender_binary'] = df['gender'].astype(str).str.strip().map({'Female': 0, 'Male': 1}).fillna(0).astype(int)

    # 7. Create composite risk score (0 to 3)
    df['mental_health_risk_score'] = df['depression'] + df['anxiety'] + df['panic_attack']
    df['high_risk_flag'] = (df['mental_health_risk_score'] >= 2).astype(int)

    return df

def save_cleaned_data(df: pd.DataFrame, output_path: str):
    """Save cleaned dataset to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved successfully to {output_path} (Shape: {df.shape})")

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(BASE_DIR, 'dataset', 'Student Mental health.csv')
    cleaned_path = os.path.join(BASE_DIR, 'dataset', 'cleaned_student_mental_health.csv')
    raw_df = load_data(raw_path)
    clean_df = clean_data(raw_df)
    save_cleaned_data(clean_df, cleaned_path)
