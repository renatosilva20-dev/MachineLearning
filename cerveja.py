# %%
import pandas as pd
df = pd.read_excel("data/dados_cerveja.xlsx")
df.head()
# %%
features = ["temperatura","copo","espuma","cor"]
target = "classe"
X = df[features]
y = df[target]

X.replace({
    "mud": 1, "pint": 2,
    "sim": 1, "nao": 0,
    "clara": 0, "escura": 1,
})

X
# %%
from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(X=X, y=y)
# %%
