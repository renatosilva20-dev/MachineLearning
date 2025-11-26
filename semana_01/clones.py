# %%
import pandas as pd
df = pd.read_parquet("data/dados_clones.parquet")
df.head()
# %%
from sklearn import tree
tree = tree.DecisionTreeClassifier()
# %%
