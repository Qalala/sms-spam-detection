# sms_spam_tk.py
# Desktop Tkinter UI for SMS Spam Detection
# Models:
#   1) Custom SVM + TF-IDF (80/20 train/test on dataset)
#   2) Pretrained Hugging Face DistilBERT (fine-tuned SMS spam detection)
#
# Supports CSV labels: 0/1, spam/ham, LABEL_0/LABEL_1

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# -----------------------
# Load Pretrained Model
# -----------------------
def load_pretrained():
    model_id = "mariagrandury/distilbert-base-uncased-finetuned-sms-spam-detection"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    return TextClassificationPipeline(model=model, tokenizer=tokenizer)

pretrained_pipe = load_pretrained()

def pretrained_predict(texts):
    outputs = pretrained_pipe(texts)
    preds = []
    for out in outputs:
        out_list = [out] if isinstance(out, dict) else out
        best_label = max(out_list, key=lambda x: x['score'])['label']
        preds.append(1 if str(best_label).lower() in ["label_1", "spam", "1"] else 0)
    return np.array(preds, dtype=int)

# -----------------------
# Custom Model (SVM+TFIDF)
# -----------------------
class SVMTextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95)
        self.model = LinearSVC()
        self.trained = False

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.trained = True

    def predict(self, texts):
        if not self.trained:
            raise RuntimeError("Train the model first.")
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

svm_clf = SVMTextClassifier()

# -----------------------
# Evaluation
# -----------------------
def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, precision, recall, f1, cm

def show_confusion_matrix(cm):
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Not Spam","Spam"]); ax.set_yticklabels(["Not Spam","Spam"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")
    plt.show()

# -----------------------
# Tkinter UI
# -----------------------
class SpamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SMS Spam Detector")
        self.dataset = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

        frm = ttk.Frame(root, padding=10)
        frm.grid()

        ttk.Button(frm, text="Load Dataset", command=self.load_dataset).grid(column=0, row=0, padx=5)
        self.model_choice = tk.StringVar(value="Custom")
        ttk.Radiobutton(frm, text="Custom (SVM+TFIDF)", variable=self.model_choice, value="Custom").grid(column=1,row=0)
        ttk.Radiobutton(frm, text="Pretrained (BERT)", variable=self.model_choice, value="Pretrained").grid(column=2,row=0)

        ttk.Button(frm, text="Train/Evaluate", command=self.train_evaluate).grid(column=0, row=1, pady=5)

        self.metrics_text = tk.Text(frm, width=60, height=10)
        self.metrics_text.grid(column=0, row=2, columnspan=3)

        ttk.Label(frm, text="Enter SMS:").grid(column=0, row=3, sticky="w")
        self.sms_entry = tk.Entry(frm, width=50)
        self.sms_entry.grid(column=1, row=3, columnspan=2)
        ttk.Button(frm, text="Classify", command=self.classify_sms).grid(column=0, row=4, pady=5)

        self.result_var = tk.StringVar()
        ttk.Label(frm, textvariable=self.result_var, font=("Arial",12,"bold")).grid(column=1, row=4)

    # -----------------------
    # Load Dataset with normalization
    # -----------------------
    def load_dataset(self):
        file = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
        if not file: return
        try:
            df = pd.read_csv(file)
            if "sms" not in df.columns or "label" not in df.columns:
                messagebox.showerror("Error","Dataset must have 'sms' and 'label' columns")
                return

            # Normalize labels
            def normalize_label(x):
                return 1 if str(x).lower() in ["1", "spam", "label_1"] else 0

            df["label"] = df["label"].apply(normalize_label)

            self.dataset = df
            X = df["sms"].astype(str).tolist()
            y = df["label"].values

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            messagebox.showinfo("Dataset","Loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # -----------------------
    # Train/Evaluate
    # -----------------------
    def train_evaluate(self):
        if self.dataset is None:
            messagebox.showerror("Error","Load a dataset first.")
            return
        model = self.model_choice.get()
        if model == "Custom":
            svm_clf.fit(self.X_train, self.y_train)
            y_pred = svm_clf.predict(self.X_test)
        else:
            y_pred = pretrained_predict(self.X_test)
        acc, prec, rec, f1, cm = evaluate(self.y_test, y_pred)
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, f"Accuracy: {acc:.3f}\nPrecision: {prec:.3f}\nRecall: {rec:.3f}\nF1 Score: {f1:.3f}\n")
        self.metrics_text.insert(tk.END, f"Confusion Matrix:\n{cm}")
        show_confusion_matrix(cm)

    # -----------------------
    # Classify single SMS
    # -----------------------
    def classify_sms(self):
        text = self.sms_entry.get()
        if not text:
            messagebox.showwarning("Warning","Enter a message first.")
            return
        model = self.model_choice.get()
        if model == "Custom":
            if not svm_clf.trained:
                messagebox.showerror("Error","Train the custom model first.")
                return
            pred = svm_clf.predict([text])[0]
        else:
            pred = pretrained_predict([text])[0]
        self.result_var.set("SPAM" if pred==1 else "NOT SPAM")

# -----------------------
# Run App
# -----------------------
root = tk.Tk()
app = SpamApp(root)
root.mainloop()
