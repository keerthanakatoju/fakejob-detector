import pandas as pd
import re
import nltk
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import shap
import numpy as np

# ------------------ Setup ------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ------------------ Load Dataset ------------------
data = pd.read_csv("fake_job_postings.csv", encoding="latin1")
data.columns = data.columns.str.strip().str.lower()
data = data[['description', 'fraudulent']].dropna()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)
    return ' '.join(w for w in text.split() if w not in stop_words)

data['cleaned_text'] = data['description'].apply(clean_text)

X = data['cleaned_text']
y = data['fraudulent']

# ------------------ Vectorizer ------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, lr_model.predict(X_test))
print("Accuracy:", accuracy)

explainer = shap.LinearExplainer(lr_model, X_train)

# ------------------ Scam Patterns ------------------
HARD_SCAM = [
    "earn money easily", "guarantee huge profits",
    "no interview required", "instant joining",
    "no experience required"
]

MEDIUM_SCAM = [
    "telegram", "whatsapp hr", "joining fee",
    "registration fee", "typing job",
    "earn daily", "limited seats",
    "work from home and earn"
]

SOFT_SCAM = ["earn", "money", "income", "no skills"]

# ------------------ Prediction ------------------
def predict_job(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    proba = lr_model.predict_proba(vector)[0][1]

    lower = text.lower()
    scam_score = 0

    # Hard scams
    for p in HARD_SCAM:
        if p in lower:
            scam_score += 3

    # Medium scams
    for m in MEDIUM_SCAM:
        if m in lower:
            scam_score += 2

    # Soft scams
    for s in SOFT_SCAM:
        if s in lower:
            scam_score += 1

    # Apply override
    if scam_score >= 5:
        proba = max(proba, 0.9)
    elif scam_score >= 3:
        proba = max(proba, 0.75)

    # SHAP explanation
    shap_values = explainer.shap_values(vector)
    names = vectorizer.get_feature_names_out()
    arr = shap_values[0]
    idx = np.argsort(np.abs(arr))[-5:]
    words = [(names[i], arr[i]) for i in idx]

    return min(proba, 1.0), words

# ------------------ UI ------------------
root = tk.Tk()
root.title("AI Fake Job Detector")
root.geometry("800x650")
root.configure(bg="#0f172a")

TITLE = ("Segoe UI", 18, "bold")
TEXT = ("Segoe UI", 11)
BTN = ("Segoe UI", 11, "bold")

tk.Label(root, text="🤖 Fake Job Detector (Explainable AI)",
         font=TITLE, fg="white", bg="#0f172a").pack(pady=15)

tk.Label(root, text=f"Model Accuracy: {accuracy*100:.2f}%",
         font=("Segoe UI", 10), fg="#38bdf8", bg="#0f172a").pack()

text_entry = tk.Text(root, height=8, width=85, font=TEXT,
                     bg="#1e293b", fg="white", insertbackground="white")
text_entry.pack(pady=15)

progress = ttk.Progressbar(root, length=400)
progress.pack(pady=5)

confidence_label = tk.Label(root, text="", fg="white", bg="#0f172a")
confidence_label.pack()

result_label = tk.Label(root, text="Result:",
                        font=("Segoe UI", 13, "bold"),
                        bg="#0f172a", fg="white", justify="left")
result_label.pack(pady=15)

# ------------------ UI Functions ------------------
def show_result(proba, words):
    fake = proba > 0.6
    confidence = proba if fake else (1 - proba)

    progress['value'] = confidence * 100
    confidence_label.config(text=f"Confidence: {confidence*100:.1f}%")

    if fake:
        color = "#ef4444"
        text = f"❌ Fake Job ({proba*100:.2f}%)\n\n"
    else:
        color = "#22c55e"
        text = f"✅ Real Job ({(1-proba)*100:.2f}%)\n\n"

    text += "Top Influencing Words:\n"
    for w, v in words:
        text += f"{w} ({v:.3f})\n"

    result_label.config(text=text, fg=color)

def check_text():
    txt = text_entry.get("1.0", tk.END).strip()
    if not txt:
        messagebox.showwarning("Error", "Enter description")
        return
    p, w = predict_job(txt)
    show_result(p, w)

def upload_file():
    path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if not path:
        return
    with open(path, encoding="utf-8") as f:
        p, w = predict_job(f.read())
    show_result(p, w)

def clear_all():
    text_entry.delete("1.0", tk.END)
    result_label.config(text="Result:", fg="white")
    progress['value'] = 0
    confidence_label.config(text="")

# Buttons
btn_frame = tk.Frame(root, bg="#0f172a")
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Check", command=check_text,
          bg="#22c55e", fg="black", font=BTN, width=12).grid(row=0, column=0, padx=10)

tk.Button(btn_frame, text="Upload TXT", command=upload_file,
          bg="#38bdf8", fg="black", font=BTN, width=12).grid(row=0, column=1, padx=10)

tk.Button(btn_frame, text="Clear", command=clear_all,
          bg="#f59e0b", fg="black", font=BTN, width=12).grid(row=0, column=2, padx=10)

tk.Label(root,
         text=" Explainable AI Project",
         font=("Segoe UI", 9), fg="#94a3b8", bg="#0f172a").pack(side="bottom", pady=15)

root.mainloop()