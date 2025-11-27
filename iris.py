# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# %%
iris = load_iris()
X = iris.data
y = iris.target

print(X)
# %%
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# %%
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
# %%
print("Predicted labels:", accuracy_score(y_test, y_pred))
# %%
 #Acuracia = 1.0 ( 100% )
 # O modelo KNeighborsClassifier conseguiu classificar corretamente todas as amostras do conjunto de teste, resultando em uma acurácia perfeita de 100%.