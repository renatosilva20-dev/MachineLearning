#%%
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data_loader import load_data
from preprocessing import preprocess_data
from model import build_model

def main():
    data = load_data()
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()

# %%
