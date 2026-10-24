# Fake Job & Internship Detector

Streamlit interface based on the Colab notebook's job-posting features and model-selection workflow. Train with the included synthetic demo CSV or upload a labeled job-posting CSV (`fraudulent` must contain both `0` and `1`), then enter a posting to get the model's fraud score and separate rule-based warning signs.

The app compares the notebook's five classifiers with and without chi-square feature selection, selects the best by F1 then ROC-AUC, and refits it on the selected data. Training runs in the current Streamlit session and the model must be retrained after a session reset or app restart.

After training, the app shows Accuracy, Precision, Recall, F1, and ROC-AUC on a held-out part of the chosen dataset, along with the validation sample count. The per-posting model score and text-based warning signs are separate checks, so they can disagree; a low model score is not proof that a posting is legitimate.

`demo_jobs.csv` contains fictional, synthetic examples so you can try the training and prediction workflow without supplying a dataset. It is not real job-posting data, and its evaluation scores must not be interpreted as evidence of real-world accuracy. Use a representative, labeled dataset for meaningful evaluation.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create an app from this repository and select `main`.
3. Set the main file path to `app.py`, then deploy.
4. Click **Train with demo dataset** to try the app, or upload a representative labeled CSV in the sidebar for meaningful model training.

The model exists only in the active app session. It must be trained again after a session reset or app restart.
