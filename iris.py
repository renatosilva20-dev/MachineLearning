# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
import numpy as np
# %%
iris = load_iris()
X = iris.data
y = iris.target

print(X)
# %%
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# %%
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# %%
train_sizes, train_scores, test_scores = learning_curve(knn,X,y,cv=5, train_sizes=np.linspace(0.1,1.0,10))

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
# %%

print("Predicted labels:", accuracy_score(y_test, y_pred))
# %%
 #Acuracia = 1.0 ( 100% )
 # O modelo KNeighborsClassifier conseguiu classificar corretamente todas as amostras do conjunto de teste, resultando em uma acurácia perfeita de 100%.
 
 # %%
import matplotlib.pyplot as plt
plt.plot(train_sizes, train_mean, label="Acuracia do treino", color="red")
plt.plot(train_sizes, test_mean, label="Validação do teste", color="blue")
plt.xlabel("Tamanho do conjunto de treino")
plt.ylabel("Acuracia")
plt.title("Curva de Aprendizado do KNeighborsClassifier no conjunto Iris")
plt.legend()
plt.show() 
# %%
