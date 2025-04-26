import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model, scaler, and PCA transformer
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca_model.pkl")  # Make sure you saved it as 'pca_model.pkl'

# Expected columns (before encoding)
base_columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

# Title of the app
st.set_page_config(page_title="Intrusion Detection System")
st.title("🚦 Intrusion Detection System – Batch CSV Prediction")

# CSV upload
uploaded_file = st.file_uploader("📁 Upload a CSV File", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.subheader("📄 Uploaded Data Preview:")
        st.dataframe(df.head())

        # Backup original data
        original_df = df.copy()

        # Handle categorical encoding
        if 'protocol_type' in df.columns or 'service' in df.columns or 'flag' in df.columns:
            df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'], drop_first=True)

        # Ensure encoded columns match model training
        model_input_columns = scaler.feature_names_in_.tolist()
        for col in model_input_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[model_input_columns]

        # Scaling & PCA
        scaled_data = scaler.transform(df)
        pca_data = pca.transform(scaled_data)

        # Predict
        predictions = model.predict(pca_data)
        original_df['Prediction'] = predictions

        st.subheader("📊 Prediction Results")
        st.dataframe(original_df[['Prediction']].value_counts().reset_index(name='Count'))

        # 📈 Improved Prediction Distribution Plot
        st.subheader("📈 Prediction Distribution")
        prediction_counts = original_df['Prediction'].value_counts().sort_values(ascending=False)

        fig, ax = plt.subplots()
        ax.bar(prediction_counts.index.astype(str), prediction_counts.values, color='skyblue')
        ax.set_xlabel("Attack Label")
        ax.set_ylabel("Count")
        ax.set_title("Attack Prediction Count")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error occurred: {e}")
else:
    st.info("Please upload a CSV file for prediction.")
