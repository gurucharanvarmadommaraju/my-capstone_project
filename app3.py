import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# === Load Trained Artifacts ===
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca_model.pkl')
model = joblib.load('random_forest_model.pkl')

# === Rename Columns for Compatibility ===
def rename_columns(data):
    return data.rename(columns={
        'protocol': 'protocol_type',
        'service_type': 'service',
        'connection_flag': 'flag'
    })

# === Preprocess Input Data ===
def preprocess_data(df):
    df = rename_columns(df)

    # Categorical encoding (simple ordinal)
    cat_cols = ['protocol_type', 'service', 'flag']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    # Drop label if present
    if 'label' in df.columns:
        df.drop(columns='label', inplace=True)

    # Align with model training
    try:
        scaled = scaler.transform(df)
        reduced = pca.transform(scaled)
        return reduced, df  # return raw df too for adding predictions
    except Exception as e:
        print("❌ Error in preprocessing:", e)
        return None, None

# === Predict ===
def predict(model, features):
    return model.predict(features)

# === Main Function ===
def run_prediction(file_path, output_path=None):
    if not os.path.exists(file_path):
        print("🚫 File not found.")
        return

    print(f"\n📥 Reading file: {file_path}")
    df = pd.read_csv(file_path)

    features, raw_df = preprocess_data(df)

    if features is not None:
        preds = predict(model, features)
        raw_df['Prediction'] = preds

        print("\n✅ Sample Predictions:")
        print(raw_df[['Prediction']].value_counts().reset_index(name="Count"))

        if output_path:
            raw_df.to_csv(output_path, index=False)
            print(f"\n📤 Predictions saved to: {output_path}")
    else:
        print("⚠️ Preprocessing failed.")

# === Run ===
if __name__ == "__main__":
    input_csv = r"C:\Users\guruc\Downloads\archive (7)\nsl-kdd-train.csv"  # ✅ Update as needed
    output_csv = r"C:\Users\guruc\Downloads\archive (7)\nsl-kdd-predicted.csv"  # Optional output
    run_prediction(input_csv, output_csv)
