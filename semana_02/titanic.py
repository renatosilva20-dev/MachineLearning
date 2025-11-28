# %%
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree
# %%
df = sns.load_dataset("titanic")
df
# %%
df['alive'].replace({'yes':1,'no':0},inplace=True)
df['alone'].replace({True:1,False:0},inplace=True)
# %%

df['deck'].fillna('D', inplace=True)
df['age'].fillna(df['age'].median(), inplace=True)
# %%
df
# %%
X = df[['pclass','sex','age','fare','embarked']]
y = df['survived']
# %%
categorical = ['sex','embarked']
numeric = ['pclass','age','fare']
# %%
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough' ,numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ])
# %%
knn = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', KNeighborsClassifier(n_neighbors=5))])

tree = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42))])

# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
#knn.fit(X_train, y_train)
tree.fit(X_train, y_train)

# %%
#y_pred_knn = knn.predict(X_test)
y_pred_tree = tree.predict(X_test)

# %%
import matplotlib.pyplot as plt
clf = tree.named_steps['classifier']
#%%
plt.figure(figsize=(20,10))
plot_tree(clf,
          feature_names=preprocessor.get_feature_names_out(),
          class_names=['Not Survived','Survived'],
          filled=True)
plt.show()
# %%