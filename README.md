# SMS Spam Detection System

A desktop tool that classifies SMS messages as **spam** or **legitimate (ham)** using two complementary approaches:

1. A **custom SVM + TF-IDF** model you train/evaluate on local CSV datasets
2. A **pretrained DistilBERT** classifier loaded from Hugging Face (no local training required)

The stack is:
- **UI**: Python Tkinter desktop app
- **Classical ML**: scikit-learn TF-IDF + LinearSVC
- **Transformer NLP**: Hugging Face Transformers DistilBERT pipeline

This project is useful for comparing classical feature-based spam detection against a modern transformer baseline in one interactive interface.

## Tech stack
- Python 3
- scikit-learn (SVM + TF-IDF)
- DistilBERT / Hugging Face Transformers
- PyTorch
- Tkinter (desktop UI)
- pandas / NumPy / Matplotlib

## Architecture (high level)

```
de_spam.py                      [primary app]
├─ Dataset loaders (CSV)
│  └─ normalize labels: spam/ham → 1/0
├─ Custom model path
│  ├─ TfidfVectorizer (unigrams + bigrams)
│  ├─ LinearSVC
│  ├─ train / evaluate / confusion matrix
│  └─ single-message predict
├─ Pretrained model path
│  ├─ load DistilBERT pipeline
│  └─ single-message / batch-style predict
└─ Tkinter UI
   ├─ model selector
   ├─ train / evaluate controls
   └─ message input + result display

backup.py                       [alternate/backup script]
sms.csv / spam.csv              [included datasets]
about-the-project.txt           [design notes]
requirements-notes.txt          [dependency notes]
```

### Models

#### A) Custom model — SVM + TF-IDF
- **TF-IDF** converts messages into sparse numeric features.
- Typical settings used in this project:
  - `ngram_range=(1, 2)` — unigrams and bigrams
  - `min_df=2` — ignore rare tokens
  - `max_df=0.95` — ignore overly common tokens
- **LinearSVC** learns a separating hyperplane between spam and ham.
- Must be **trained on a loaded dataset** before custom predictions work.

#### B) Pretrained model — DistilBERT
- Model id: `mariagrandury/distilbert-base-uncased-finetuned-sms-spam-detection`
- Uses a Hugging Face `TextClassificationPipeline`
- Ready immediately after download/load (no local fine-tuning step)
- Output labels are mapped to binary `0/1` (ham/spam)

### Data handling
- Expected CSV columns: message text + label (commonly `sms` / `label`, with normalization in code)
- Labels are normalized so variants like `spam`, `ham`, `1`, `0` become binary targets
- Train/test split uses stratification (e.g. 80/20) so class balance is preserved

### UI behavior highlights
- Choose custom vs pretrained model
- Irrelevant buttons are disabled based on the selected model
- Custom path supports train + evaluate + confusion-matrix visualization
- Both paths support classifying a single SMS string

---

## Running the project

### Prerequisites
- Python 3.8+
- Enough disk/network for the first DistilBERT model download
- Tkinter available in your Python install

### Install dependencies

```bash
pip install scikit-learn pandas numpy matplotlib transformers torch
```

### Start the app

```bash
python de_spam.py
```

On first DistilBERT use, Transformers will download the pretrained weights from Hugging Face.

---

## How to navigate the repo
- `de_spam.py` — main Tkinter application
- `backup.py` — backup / alternate implementation
- `sms.csv`, `spam.csv` — sample/training datasets
- `about-the-project.txt` — written overview of models and flow
- `requirements-notes.txt` — dependency notes from development

---

## Demo walkthrough

### Custom SVM path
1. Launch the app.
2. Select the **custom / SVM** model option.
3. Load one of the included CSVs (or your own with compatible columns).
4. Train the model.
5. Evaluate and inspect the confusion matrix.
6. Paste a new SMS and classify it.

### DistilBERT path
1. Select the **pretrained DistilBERT** option.
2. Wait for model load (first run only).
3. Paste an SMS and classify immediately (no training step).

### Example messages to try
- Ham-like: `Hey, are we still meeting at 6?`
- Spam-like: `Congratulations! You won a free iPhone. Click this link now.`

---

## Project purpose
Provide a practical, side-by-side SMS spam detection demo that contrasts classical ML (TF-IDF + SVM) with a transformer baseline (DistilBERT) inside one desktop tool.
