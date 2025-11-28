# %%
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# %%
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)

# %%
matriz_de_confusao = pd.crosstab(y_test, y_pred,
                        rownames=['Real'],
                        colnames=['Previsto'],
                        margins=True)

print(matriz_de_confusao)
# %%
