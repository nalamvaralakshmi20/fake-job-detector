# Fake Job & Internship Detector

Streamlit interface based on the Colab notebook's job-posting features and model-selection workflow. Train with the included synthetic demo CSV or upload a labeled job-posting CSV (`fraudulent` must contain both `0` and `1`), inspect the original and model-ready data, choose a trained model, then enter a posting to get its score and separate rule-based warning signs.

The app compares five classifiers—with and without chi-square feature selection—using stratified 5-fold out-of-fold predictions (or fewer folds when the smallest class has fewer than five examples). Every labeled row is predicted by a model that did not train on it. The app displays each configuration's Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, and Specificity. The best out-of-fold F1 (ROC-AUC breaks ties) is selected initially, but you can choose any trained configuration for analysis. Since the app chooses the top configuration from those same folds, use a separate untouched test set for an unbiased final model-performance estimate.

After training, the app shows the uploaded raw CSV and a cleaned/engineered ETL preview, with download buttons for both. The searchable posting selector includes every valid labeled CSV row, not only a small test subset. Select a row to inspect its out-of-fold score and known label, or load its fields into the manual analysis form. Training data, model, and analysis result are held in the current Streamlit browser session, are not shared with other app sessions, and must be retrained after a session reset or app restart.

The cleaned preview normalizes text (including URL/email tokens), fills missing categorical values, normalizes boolean flags and labels, and displays derived features. The model also uses location and salary fields as categorical inputs. A manual score comes from the final model refit on all labeled rows; if the posting came from the CSV, use its out-of-fold score for an independent check. Every manual result keeps the selected model's percentage alongside separate rule-based warnings and missing details. These checks can disagree; a low model score is not proof that a posting is legitimate.

`demo_jobs.csv` contains 24 fictional, synthetic examples so you can try the training and prediction workflow without supplying a dataset. Because the examples are small and deliberately constructed, all models can score 100% even under out-of-fold evaluation. That result only describes this toy dataset; it is not credible evidence of real-world accuracy. Use a much larger, representative dataset with trustworthy labels and a final independent test set for meaningful evaluation.

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
4. Click **Train with demo dataset** to try the app, or upload a representative labeled CSV in the sidebar for meaningful model training. After training, compare the model scores and select a model before analyzing postings.

The model exists only in the active app session. It must be trained again after a session reset or app restart.
