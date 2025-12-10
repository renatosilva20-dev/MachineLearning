# %%
import pandas as pd 
from sklearn.model_selection import train_test_split
# %%
df = pd.read_csv('data/spam.csv', encoding='latin-1')
df
# %%
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
# %%
df['label'] = df['label'].map({'ham':0, 'spam':1})
# %%
df
# %%
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
# %%
X = df['message']
y = df['label']
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# %%
vec = CountVectorizer()
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)
# %%
model = MultinomialNB()
model.fit(X_train_vec, y_train)
# %%
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of predict the Spam: {accuracy:.2f}')
# %%
