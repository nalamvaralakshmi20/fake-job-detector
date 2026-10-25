# Fake Job & Internship Detector

Streamlit interface based on the Colab notebook's job-posting features and model-selection workflow. Train with the included synthetic demo CSV or upload a labeled job-posting CSV (`fraudulent` must contain both `0` and `1`), inspect the original and model-ready data, choose a trained model, then enter a posting to get its score and separate rule-based warning signs.

The app separately trains and evaluates five classifiers—with and without chi-square feature selection—and displays each configuration's Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, and Specificity. The best validation F1 (ROC-AUC breaks ties) is selected initially, but you can choose any trained configuration for analysis. Each result remains visible when Streamlit reruns and is labeled with the model that produced it.

After training, the app shows the uploaded raw CSV and a cleaned/engineered ETL preview, with download buttons for both. It also lets you choose a held-out validation row and compare its known label with the selected model's prediction and fraud score. Only held-out rows are offered, so the row-level example is not scored by a model trained on that row. Training data, model, and analysis result are held in the current Streamlit browser session, are not shared with other app sessions, and must be retrained after a session reset or app restart.

The cleaned preview normalizes text (including URL/email tokens), fills missing categorical values, normalizes boolean flags and labels, and displays derived features. The model also uses location and salary fields as categorical inputs. Each manual posting result keeps the selected model's percentage alongside a separate count of rule-based warning signs and missing details. These checks can disagree; a low model score is not proof that a posting is legitimate. If the app updates while open, retrain when prompted to refresh an older cached model.

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
4. Click **Train with demo dataset** to try the app, or upload a representative labeled CSV in the sidebar for meaningful model training. After training, compare the model scores and select a model before analyzing postings.

The model exists only in the active app session. It must be trained again after a session reset or app restart.
