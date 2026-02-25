#%%
import pandas as pd

def load_data(path="creditcard.csv"):
    return pd.read_csv(path)

# %%
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(data):
    X = data.drop("Class", axis=1)
    y = data["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return X_resampled, y_resampled

# %%
