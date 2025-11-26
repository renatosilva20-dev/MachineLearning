# %%
import pandas as pd

df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df.head()

df['aprovado'] = (df['nota']>5).astype(int)
df
# %%

import matplotlib.pyplot as plt 
plt.plot(df['cerveja'],df['aprovado'],'o', color='royalblue')
plt.grid(True)
plt.title("Cerveja vs Aprovado")
plt.xlabel("Cerveja")
plt.ylabel("Aprovado")

# %%
from sklearn import linear_model
from sklearn import tree
from sklearn import naive_bayes

reg = linear_model.LogisticRegression(penalty=None,
                                      fit_intercept=True)
reg.fit(df[['cerveja']], df['aprovado'])
reg_predict = reg.predict(df[['cerveja']].drop_duplicates())
reg_prob = reg.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

arvore_full = tree.DecisionTreeClassifier(random_state=42)
arvore_full.fit(df[['cerveja']],df['aprovado'])
arvore_predict = arvore_full.predict(df[['cerveja']].drop_duplicates())
arvore_proba = arvore_full.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

nb = naive_bayes.GaussianNB()
nb.fit(df[['cerveja']],df['aprovado'])
nb_predict = nb.predict(df[['cerveja']].drop_duplicates())
nb_proba = nb.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

plt.figure(dpi=400)
plt.plot(df['cerveja'],df['aprovado'],'o', color='royalblue')
plt.grid(True)
plt.title("Cerveja vs Aprovado")
plt.xlabel("Cerveja")
plt.ylabel("Aprovado")
plt.hlines(0.5,xmin=1,xmax=9,linestyles='--',color='black')
plt.plot(df['cerveja'].drop_duplicates(), reg_predict, color='red')
plt.plot(df['cerveja'].drop_duplicates(), reg_prob, color='orange')
plt.plot(df['cerveja'].drop_duplicates(), arvore_predict, color='purple')
plt.plot(df['cerveja'].drop_duplicates(), arvore_proba, color='green')
plt.plot(df['cerveja'].drop_duplicates(), nb_predict, color='gray')
plt.plot(df['cerveja'].drop_duplicates(), nb_proba, color='blue')

plt.legend(['Observação',
           'Reg Predict',
           'Reg Proba',
           'Arvore Predict',
           'Arvore Prob',
           'Naive Bayes Predict',
           'Naive Bayes Proba'])
# %%
