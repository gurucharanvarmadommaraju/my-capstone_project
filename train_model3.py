# train_model.py (for UNSW-NB15)
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import joblib

# Step 1: Load the Dataset (UNSW-NB15 CSV)
df = pd.read_csv("UNSW_NB15_testing-set.csv")

# Step 2: Encode Categorical Columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Step 3: Define Features (X) and Target (y)
# Usually, the target is a column named 'label' or 'attack_cat' — confirm this
# If your file uses 'label' where 0 = normal, 1 = attack, use this:
X = df.drop("label", axis=1)
y = df["label"]

# Step 4: Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply PCA (retain 95% variance)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Step 6: Split into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Step 7: Train the Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Save Model, PCA, Scaler
joblib.dump(model, "rf_model1_unsw.pkl")
joblib.dump(scaler, "scaler1_unsw.pkl")
joblib.dump(pca, "pca1_unsw.pkl")

print("✅ Model, PCA, and Scaler saved successfully for UNSW-NB15.")
