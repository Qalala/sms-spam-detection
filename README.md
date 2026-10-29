# SMS Spam Detection System

## Brief description
A desktop tool that classifies SMS messages as spam or legitimate using two models: a custom **SVM + TF-IDF** pipeline and a pretrained **DistilBERT** classifier, with training and evaluation built in.

## Project purpose
Compare classical ML and transformer-based NLP for SMS spam detection, with a simple Tkinter UI for interactive classification and model evaluation.

## Project requirements
- Python 3.8+
- Packages listed below (Tkinter is usually included on Windows)

```bash
pip install scikit-learn pandas numpy matplotlib transformers torch
```

Notes:
- First DistilBERT run downloads the model from Hugging Face (`mariagrandury/distilbert-base-uncased-finetuned-sms-spam-detection`).
- Dataset files `sms.csv` / `spam.csv` are included.

## How to navigate
- `de_spam.py` — primary application
- `backup.py` — alternate/backup script
- `sms.csv`, `spam.csv` — training/evaluation datasets
- `about-the-project.txt` — original project notes
- `requirements-notes.txt` — dependency notes from development

## How to run

```bash
python de_spam.py
```

Use the GUI to train/evaluate the SVM model and/or classify messages with DistilBERT.
