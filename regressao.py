# %%
import pandas as pd

df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df.head()
# %%
from sklearn import linear_model
from sklearn import tree

X = df[['cerveja']]
y = df['nota']
# Isso é o aprendizado de maquina
reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(X, y)

arvore_d2 = tree.DecisionTreeRegressor(random_state=42,max_depth=2)
arvore_d2.fit(X, y)

predict_arvore_d2 = arvore_d2.predict(X.drop_duplicates())
# %%
a,b = reg.intercept_, reg.coef_[0]
print(a,b)
# %%
predict_reg = reg.predict(X.drop_duplicates())
predict_reg
# %%
import matplotlib.pyplot as plt
plt.plot(X['cerveja'],y,'o')
plt.grid(True)
plt.title("Relação Cerveja vs Nota")
plt.xlabel("Cerveja")
plt.ylabel("Nota")

plt.plot(X.drop_duplicates()['cerveja'],predict_reg)
plt.plot(X.drop_duplicates()['cerveja'],predict_arvore_full)
plt.plot(X.drop_duplicates()['cerveja'],predict_arvore_d2)

plt.plot(X.drop_duplicates()['cerveja'],predict_reg)
plt.legend(['Observado',f'y = {a:.3f} + {b:.3f} x',
            'ArvoreFull',
            'Arvore Depht 2'])
# %%

