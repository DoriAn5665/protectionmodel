import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI-аналіз загроз у хмарі", layout="wide")
st.title("🔐 Аналіз загроз у хмарних середовищах за допомогою AI")

st.header("1️⃣ Теплова карта активності користувачів")
user_ids = [f'User_{i}' for i in range(1, 11)]
hours = list(range(24))
activity_data = np.random.poisson(lam=5, size=(len(user_ids), len(hours)))
activity_df = pd.DataFrame(activity_data, index=user_ids, columns=hours)

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(activity_df, cmap='YlOrRd', linewidths=0.5, ax=ax)
st.pyplot(fig)

st.header("2️⃣ Виявлення аномальної активності")
summary_df = pd.DataFrame(activity_data.sum(axis=1), index=user_ids, columns=['total_requests'])
summary_df.loc['User_3', 'total_requests'] = 150  # Вставка аномалії

model = IsolationForest(contamination=0.1)
summary_df['anomaly'] = model.fit_predict(summary_df[['total_requests']])
summary_df['anomaly'] = summary_df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.barplot(data=summary_df.reset_index(), x='index', y='total_requests', hue='anomaly', palette={'Normal': 'green', 'Anomaly': 'red'}, ax=ax2)
ax2.set_ylabel('Запитів за добу')
ax2.set_xlabel('Користувач')
st.pyplot(fig2)

st.header("3️⃣ Класифікація загроз за поведінковими ознаками")
st.write("**Ознаки:** 1 — внутрішній користувач, 2 — шифрування ввімкнене")
X = [[1, 0], [0, 1], [1, 1], [0, 0], [1, 1], [0, 0], [1, 0], [0, 1]]
y = [1, 1, 1, 0, 1, 0, 0, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

example = [st.slider('Внутрішній користувач (0/1)', 0, 1, 1),
           st.slider('Шифрування (0/1)', 0, 1, 1)]
pred = clf.predict([example])[0]

st.success(f"Ймовірність загрози: {'Висока' if pred == 1 else 'Низька'}")
