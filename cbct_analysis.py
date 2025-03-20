import os
import sqlite3
import pandas as pd
import pydicom
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import streamlit as st

# Create data storage directory
DATA_DIR = "cbct_condylography_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Database initialization
DB_PATH = "cbct_analysis.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create tables for data storage
cursor.execute('''
CREATE TABLE IF NOT EXISTS CBCT_Scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT,
    file_path TEXT,
    date_uploaded TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS Condylography_Records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT,
    file_path TEXT,
    date_uploaded TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS Research_Studies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Authors TEXT,
    Publication_Date TEXT,
    Abstract TEXT,
    Link TEXT
)
''')
conn.commit()

# Store CBCT files
def store_cbct(patient_id, file_path):
    cursor.execute("INSERT INTO CBCT_Scans (patient_id, file_path) VALUES (?, ?)", (patient_id, file_path))
    conn.commit()

# Store condylography records
def store_condylography(patient_id, file_path):
    cursor.execute("INSERT INTO Condylography_Records (patient_id, file_path) VALUES (?, ?)", (patient_id, file_path))
    conn.commit()

# AI Model for CBCT Analysis
def build_cbct_ai_model(input_shape=(256, 256, 1)):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Streamlit dashboard
def research_dashboard():
    st.set_page_config(page_title="CBCT & Condylography Analysis", layout="wide")
    st.title("🔍 CBCT, Condylography, and Research Insights")

    conn = sqlite3.connect(DB_PATH)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    
    table_selection = st.selectbox("📌 Select Data Type:", tables['name'])
    df = pd.read_sql(f"SELECT * FROM {table_selection}", conn)
    st.write("### 📜 Data Overview")
    st.dataframe(df)
    
    conn.close()

if __name__ == "__main__":
    research_dashboard()
