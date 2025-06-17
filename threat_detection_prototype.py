# threat_detection_prototype.py

import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import smtplib
from email.mime.text import MIMEText

# === 1. ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ === #
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.fillna(0, inplace=True)
    return df

def preprocess(df):
    agg = df.groupby('user_id').agg({
        'request_count': 'sum',
        'timestamp': ['min', 'max'],
        'ip': pd.Series.nunique
    })
    agg.columns = ['request_count', 'first_seen', 'last_seen', 'unique_ips']
    return agg.reset_index()

# === 2. ВИЯВЛЕННЯ ЗАГРОЗ === #
def detect_anomalies(data):
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    features = data[['request_count', 'unique_ips']]
    model.fit(features)
    data['threat'] = model.predict(features)
    data['threat_label'] = data['threat'].map({1: 'normal', -1: 'anomaly'})
    return data

# === 3. РЕАГУВАННЯ === #
def send_alert(user_id):
    msg = MIMEText(f"Увага! Виявлено підозрілу активність користувача: {user_id}")
    msg['Subject'] = 'Інцидент безпеки'
    msg['From'] = 'security@system.local'
    msg['To'] = 'admin@company.com'

    try:
        with smtplib.SMTP('smtp.example.com') as server:
            server.send_message(msg)
    except Exception as e:
        print(f"Помилка надсилання: {e}")

# === 4. STREAMLIT ІНТЕРФЕЙС === #
st.title("Прототип системи виявлення та реагування на загрози")

uploaded_file = st.file_uploader("Завантажте лог-файл (CSV)", type=["csv"])

if uploaded_file:
    logs = load_data(uploaded_file)
    processed = preprocess(logs)
    result = detect_anomalies(processed)

    st.subheader("Результати аналізу")
    st.dataframe(result)

    st.subheader("Візуалізація активності")
    fig, ax = plt.subplots()
    sns.histplot(data=result, x='request_count', hue='threat_label', multiple='stack', ax=ax)
    st.pyplot(fig)

    st.subheader("Виявлені аномалії")
    anomalies = result[result['threat_label'] == 'anomaly']
    st.dataframe(anomalies)

    if st.button("Надіслати сповіщення про інциденти"):
        for uid in anomalies['user_id']:
            send_alert(uid)
        st.success("Сповіщення надіслано!")
