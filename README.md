# Fake Job & Internship Detector

Streamlit interface based on the Colab notebook's job-posting features and model-selection workflow. Upload a labeled job-posting CSV (`fraudulent` must contain both `0` and `1`), train the model, then enter a posting to get its estimated fraud probability and rule-based red flags.

The app compares the notebook's five classifiers with and without chi-square feature selection, selects the best by F1 then ROC-AUC, and refits it on the uploaded data. The dataset and trained model are not committed; training runs in the current Streamlit session.

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
4. Upload the labeled CSV in the app sidebar to train the model.

The model exists only in the active app session. It must be trained again after a session reset or app restart.
