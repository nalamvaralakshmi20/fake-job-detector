import re
from pathlib import Path

import streamlit as st
import pandas as pd
import html as _html
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from modeling import (
    MODEL_ASSET_VERSION,
    add_features,
    train_best_model,
)

DEMO_DATA_PATH = Path(__file__).with_name("demo_jobs.csv")
BINARY_FORM_FIELDS = {"telecommuting", "has_company_logo", "has_questions"}
REQUIRED_MODEL_ASSET_FIELDS = {
    "asset_version",
    "models",
    "model_scores",
    "validation_predictions",
    "confusion_counts",
    "metrics",
    "selected_model",
    "training_rows",
    "validation_rows",
    "validation_folds",
}
def calculate_metrics(y_true, predictions, probabilities):
    has_both_classes = len(set(y_true)) == 2
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    metrics = {
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced accuracy": (
            balanced_accuracy_score(y_true, predictions)
            if has_both_classes
            else None
        ),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": (
            roc_auc_score(y_true, probabilities)
            if has_both_classes
            else None
        ),
        "Specificity": tn / (tn + fp) if tn + fp else None,
    }
    confusion_counts = {
        "True negative": int(tn),
        "False positive": int(fp),
        "False negative": int(fn),
        "True positive": int(tp),
    }
    return metrics, confusion_counts


def model_assets_are_compatible(assets):
    if not isinstance(assets, dict):
        return False
    if not REQUIRED_MODEL_ASSET_FIELDS.issubset(assets):
        return False
    if assets.get("asset_version") != MODEL_ASSET_VERSION:
        return False
    models = assets["models"]
    scores = assets["model_scores"]
    predictions = assets["validation_predictions"]
    confusion = assets["confusion_counts"]
    if not all(
        isinstance(value, dict)
        for value in (models, scores, predictions, confusion)
    ):
        return False
    if not models or not set(models).issubset(scores):
        return False
    if not set(models).issubset(predictions) or not set(models).issubset(confusion):
        return False
    if assets["selected_model"] not in models:
        return False
    return all(
        isinstance(assets.get(field), int) and assets[field] > 0
        for field in ("training_rows", "validation_rows", "validation_folds")
    )

# Configure page style
st.set_page_config(
    page_title="Fake Job & Internship Detector",
    page_icon="🛡️",
    layout="wide"
)

# Custom high-contrast CSS styled for Streamlit
st.markdown("""
<style>
.stApp {
    background: radial-gradient(ellipse at top left, #dbeafe 0%, transparent 45%),
                linear-gradient(135deg, #eff6ff 0%, #f5f3ff 50%, #fdf2f8 100%);
}
[data-testid="stMain"] {
    color: #111827;
}
[data-testid="stMain"] [data-testid="stWidgetLabel"],
[data-testid="stMain"] [data-testid="stWidgetLabel"] p,
[data-testid="stMain"] .stMarkdown,
[data-testid="stMain"] .stMarkdown p,
[data-testid="stMain"] .stMarkdown h3 {
    color: #111827 !important;
}
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 12px 14px;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div,
[data-testid="stMetricDelta"] {
    color: #111827 !important;
}
[data-testid="stAlert"] {
    background: #fff7cc !important;
    border: 1px solid #eab308 !important;
}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
    color: #422006 !important;
}
[data-testid="stMain"] input,
[data-testid="stMain"] textarea,
[data-testid="stMain"] [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #111827 !important;
    border-color: #94a3b8 !important;
}
[data-testid="stMain"] input::placeholder,
[data-testid="stMain"] textarea::placeholder {
    color: #64748b !important;
    opacity: 1;
}
[data-testid="stMain"] [data-baseweb="select"] svg {
    fill: #374151 !important;
}
.fj-hero {
    background: linear-gradient(120deg, #1e3a8a, #6d28d9 60%, #db2777);
    color: #ffffff !important;
    padding: 26px 30px;
    border-radius: 16px;
    margin-bottom: 24px;
}
.fj-hero h1 {
    margin: 0 0 6px;
    font-size: 32px;
    color: #ffffff !important;
    font-weight: 800;
}
.fj-hero p {
    margin: 0;
    font-size: 16px;
    color: #ffffff !important;
}
.fj-pills span {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    color: #ffffff !important;
    padding: 5px 12px;
    border-radius: 20px;
    margin: 10px 6px 0 0;
    font-size: 13px;
    font-weight: 600;
}
.fj-card {
    background: #ffffff !important;
    border: 1px solid #cbd5e1;
    border-radius: 14px;
    padding: 24px;
    color: #111827 !important;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
}
.fj-card h3 {
    color: #111827 !important;
    font-size: 20px;
    font-weight: 700;
    margin-top: 22px;
    margin-bottom: 12px;
}
.fj-top {
    display: flex;
    gap: 24px;
    align-items: center;
    flex-wrap: wrap;
}
.fj-gauge {
    width: 140px;
    height: 140px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    background: conic-gradient(
        var(--c) calc(var(--p) * 1%),
        #d1d5db 0
    );
}
.fj-gauge-inner {
    width: 108px;
    height: 108px;
    border-radius: 50%;
    background: #ffffff !important;
    color: #111827 !important;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.fj-gauge-inner b {
    color: #111827 !important;
    font-size: 26px;
    font-weight: 800;
}
.fj-gauge-inner small {
    color: #374151 !important;
    font-size: 11px;
}
.fj-badge {
    display: inline-block;
    color: #ffffff !important;
    padding: 8px 14px;
    border-radius: 20px;
    font-weight: 700;
    text-align: center;
}
.fj-risk-text {
    color: #1f2937 !important;
    font-size: 15px;
    line-height: 1.6;
    margin-top: 8px;
}
.fj-flag {
    background: #fff7ed !important;
    border-left: 5px solid #ea580c;
    padding: 10px 12px;
    margin: 8px 0;
    border-radius: 7px;
    font-size: 15px;
    color: #7c2d12 !important;
}
.fj-signal-summary {
    margin-top: 12px;
    color: #374151 !important;
    font-size: 14px;
    font-weight: 600;
}
.fj-info {
    background: #eff6ff !important;
    border-left: 5px solid #2563eb;
    padding: 10px 12px;
    margin: 8px 0;
    border-radius: 7px;
    font-size: 15px;
    color: #1e3a8a !important;
}
.fj-ok {
    background: #ecfdf5 !important;
    border-left: 5px solid #059669;
    padding: 10px 12px;
    border-radius: 7px;
    font-size: 15px;
    color: #065f46 !important;
}
.fj-chip {
    display: inline-block;
    padding: 6px 12px;
    margin: 4px 5px 4px 0;
    border-radius: 18px;
    font-size: 14px;
    font-weight: 600;
}
.fj-red {
    background: #fee2e2 !important;
    color: #991b1b !important;
}
.fj-green {
    background: #dcfce7 !important;
    color: #166534 !important;
}
.fj-tip {
    margin-top: 18px;
    font-size: 13px;
    line-height: 1.6;
    color: #374151 !important;
}
.fj-foot {
    text-align: center;
    color: #374151 !important;
    font-size: 13px;
    margin-top: 32px;
}
.fj-foot span {
    display: inline-block;
    background: #e0e7ff !important;
    color: #312e81 !important;
    padding: 5px 12px;
    border-radius: 12px;
    margin: 4px;
}
</style>
""", unsafe_allow_html=True)

if "model_assets" not in st.session_state:
    st.session_state.model_assets = None

if "model_assets_outdated" not in st.session_state:
    st.session_state.model_assets_outdated = False

if "training_data_raw" not in st.session_state:
    st.session_state.training_data_raw = None

if "training_data_cleaned" not in st.session_state:
    st.session_state.training_data_cleaned = None

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "csv_analysis_result" not in st.session_state:
    st.session_state.csv_analysis_result = None

if "selected_csv_rows" not in st.session_state:
    st.session_state.selected_csv_rows = []

assets = st.session_state.model_assets
if assets is not None and not model_assets_are_compatible(assets):
    st.session_state.model_assets = None
    st.session_state.analysis_result = None
    st.session_state.csv_analysis_result = None
    st.session_state.selected_csv_rows = []
    st.session_state.model_assets_outdated = True

with st.sidebar:
    st.header("Train the detector")
    st.caption(
        "Upload a labeled CSV or train with the included synthetic demo. "
        "Your CSV needs a `fraudulent` column with both 0 (genuine) and "
        "1 (fraudulent) labels."
    )
    training_file = st.file_uploader("Labeled job-posting CSV", type=["csv"])
    upload_train_clicked = st.button(
        "Train with uploaded CSV", type="primary", disabled=training_file is None
    )
    demo_train_clicked = st.button("Train with demo dataset")
    st.download_button(
        "Download demo CSV",
        data=DEMO_DATA_PATH.read_bytes(),
        file_name=DEMO_DATA_PATH.name,
        mime="text/csv",
    )
    st.markdown("#### What the model does")
    st.caption(
        "It trains Decision Tree, Naive Bayes, Logistic Regression, SVM, "
        "and Random Forest separately. Compare their validation scores and "
        "choose which one analyzes a posting."
    )
    if demo_train_clicked or (upload_train_clicked and training_file is not None):
        try:
            training_data = (
                pd.read_csv(DEMO_DATA_PATH)
                if demo_train_clicked
                else pd.read_csv(training_file)
            )
            with st.spinner(
                "Comparing 10 model configurations with stratified "
                "cross-validation, then fitting them on all labeled rows..."
            ):
                st.session_state.model_assets = train_best_model(training_data)
                st.session_state.model_assets["training_source"] = (
                    "Synthetic demo" if demo_train_clicked else training_file.name
                )
                st.session_state.training_data_raw = training_data.copy()
                cleaned_training_data = add_features(training_data)
                cleaned_training_data["fraudulent"] = pd.to_numeric(
                    cleaned_training_data["fraudulent"], errors="coerce"
                )
                cleaned_training_data = cleaned_training_data.dropna(
                    subset=["fraudulent"]
                )
                cleaned_training_data["fraudulent"] = (
                    cleaned_training_data["fraudulent"].astype(int)
                )
                st.session_state.training_data_cleaned = cleaned_training_data
                st.session_state.analysis_result = None
                st.session_state.csv_analysis_result = None
                st.session_state.model_choice = st.session_state.model_assets[
                    "selected_model"
                ]
                st.session_state.model_assets_outdated = False
        except Exception as error:
            st.error(f"Model training failed: {error}")

    if st.session_state.model_assets_outdated:
        st.info("The app was updated. Train the model again to refresh its scores.")

assets = st.session_state.model_assets
selected_model = None
if assets is None:
    st.warning(
        "Train with the included demo dataset or upload your own labeled CSV "
        "in the sidebar before analyzing a job posting. The model is kept "
        "for this app session."
    )
else:
    st.sidebar.success(
        f"{len(assets['models'])} models ready · "
        f"best cross-validation F1 {assets['metrics']['F1']:.3f}"
    )

# Hero Section
st.markdown(f"""
<div class="fj-hero">
  <h1>🛡️ Fake Job &amp; Internship Detector</h1>
  <p>Enter a job posting to check for possible scam patterns using Machine Learning.</p>
  <div class="fj-pills">
    <span>🤖 Trained from labeled examples</span>
    <span>🚩 Rule-based red flags</span>
  </div>
</div>
""", unsafe_allow_html=True)

if assets is not None:
    model_options = list(assets["models"])
    if st.session_state.get("model_choice") not in model_options:
        st.session_state.model_choice = assets["selected_model"]
    selected_model = st.selectbox(
        "Model used for analysis",
        model_options,
        key="model_choice",
        help="Each option is a separately trained model with its own validation scores.",
    )
    selected_metrics = assets["model_scores"][selected_model]
    st.markdown("#### Selected model's cross-validation scores")
    metric_columns = st.columns(5)
    for column, metric_name in zip(
        metric_columns,
        ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC"),
    ):
        column.metric(metric_name, f"{selected_metrics[metric_name]:.1%}")
    st.caption(
        f"Selected model: {selected_model} · "
        f"training rows: {assets['training_rows']:,} · "
        f"out-of-fold rows: {assets['validation_rows']:,} across "
        f"{assets.get('validation_folds', 'unknown')} stratified folds · "
        f"source: {assets['training_source']}. Every row is evaluated by a "
        "fold model that did not train on that row."
    )
    score_rows = [
        {"Model configuration": name, **metrics}
        for name, metrics in assets["model_scores"].items()
    ]
    score_table = pd.DataFrame(score_rows).set_index("Model configuration")
    st.dataframe(
        score_table.style.format("{:.1%}"),
        width="stretch",
    )
    st.caption(
        "These are out-of-fold scores for comparing model configurations, "
        "not a guarantee of future performance. The top configuration is "
        "chosen from these same folds, so use a separate untouched test set "
        "for an unbiased final estimate. Synthetic demo results cannot "
        "establish real-world accuracy."
    )
    with st.expander("More validation detail: balanced accuracy, specificity, confusion counts"):
        selected_confusion = assets["confusion_counts"][selected_model]
        confusion_columns = st.columns(4)
        for column, label in zip(confusion_columns, selected_confusion):
            column.metric(label, selected_confusion[label])
        st.caption(
            "Specificity measures how often genuine postings are correctly "
            "identified. False positives are genuine jobs flagged as fake; "
            "false negatives are fake jobs missed by the model."
        )
    if assets["training_rows"] < 1000 or assets["validation_rows"] < 100:
        st.warning(
            "This dataset is small, so cross-validation scores can vary "
            "substantially with different data. The demo is synthetic; even "
            "100% across these folds is not evidence of real-world accuracy. "
            "Use a large, representative, independently labeled dataset."
        )

if st.session_state.training_data_raw is not None:
    raw_data = st.session_state.training_data_raw
    cleaned_data = st.session_state.training_data_cleaned
    st.subheader("Training data loaded for this session")
    class_counts = pd.to_numeric(
        raw_data["fraudulent"], errors="coerce"
    ).value_counts()
    count_columns = st.columns(3)
    count_columns[0].metric("Rows loaded", f"{len(raw_data):,}")
    count_columns[1].metric("Genuine labels (0)", f"{int(class_counts.get(0, 0)):,}")
    count_columns[2].metric(
        "Fraudulent labels (1)", f"{int(class_counts.get(1, 0)):,}"
    )
    raw_tab, cleaned_tab = st.tabs(["Raw uploaded CSV", "Cleaned + engineered data"])
    with raw_tab:
        st.caption(
            "Original CSV values, as loaded. This data is held in this browser "
            "session and is not shared with other app sessions."
        )
        st.dataframe(raw_data, width="stretch", height=320)
        st.download_button(
            "Download raw data",
            data=raw_data.to_csv(index=False).encode("utf-8"),
            file_name="training_data_raw.csv",
            mime="text/csv",
            key="download_raw_training_data",
        )
    with cleaned_tab:
        st.caption(
            "ETL output used by the models: text normalized to lowercase, "
            "URLs/emails tokenized, missing categories marked unknown, "
            "boolean flags normalized, and numeric/text features derived. "
            "Original values remain available in the raw tab."
        )
        st.dataframe(cleaned_data, width="stretch", height=320)
        st.download_button(
            "Download cleaned + engineered data",
            data=cleaned_data.to_csv(index=False).encode("utf-8"),
            file_name="training_data_cleaned.csv",
            mime="text/csv",
            key="download_cleaned_training_data",
        )

if assets is not None and st.session_state.training_data_raw is not None:
    st.subheader("Select CSV postings to analyze")
    st.caption(
        "Each CSV row is one posting. The table's leftmost index starts at 0, "
        "so its first posting appears here as “Data row 0 / CSV line 2” "
        "(line 1 is the header). Choose one or more labeled rows. "
        "Scores use out-of-fold predictions, not the final model fitted on "
        "those same rows."
    )
    validation_rows = assets["validation_predictions"][selected_model]
    raw_data = st.session_state.training_data_raw
    row_labels = {}
    for prediction in validation_rows:
        source_row = prediction["row_number"] - 2
        raw_row = raw_data.iloc[source_row]
        title_value = str(raw_row.get("title", "")).strip() or "Untitled posting"
        row_labels[
            f"Data row {source_row} / CSV line {prediction['row_number']} · "
            f"{title_value[:90]}"
        ] = prediction

    if row_labels:
        selected_row_labels = st.multiselect(
            "CSV postings to analyze",
            list(row_labels),
            key="selected_csv_rows",
            placeholder=(
                "Select Data row 0 / CSV line 2 · "
                "Remote data entry assistant"
            ),
        )
        action_columns = st.columns(2)
        evaluate_rows_clicked = action_columns[0].button(
            "Analyze selected CSV rows",
            disabled=not selected_row_labels,
        )
        if selected_row_labels:
            row_to_load = (
                selected_row_labels[0]
                if len(selected_row_labels) == 1
                else st.selectbox(
                    "Choose which selected posting to load into the manual form",
                    selected_row_labels,
                    key="csv_row_to_load",
                )
            )
        else:
            row_to_load = None
        load_row_clicked = action_columns[1].button(
            "Load selected posting into manual form",
            disabled=row_to_load is None,
            help=(
                "A single manual form represents one job. For multiple "
                "selected postings, choose which one to load here."
            ),
        )
        if evaluate_rows_clicked:
            selected_predictions = [
                row_labels[label] for label in selected_row_labels
            ]
            y_true = [item["actual_label"] for item in selected_predictions]
            y_pred = [item["predicted_label"] for item in selected_predictions]
            y_probability = [
                item["fraud_probability"] for item in selected_predictions
            ]
            row_metrics, row_confusion = calculate_metrics(
                y_true, y_pred, y_probability
            )
            st.session_state.csv_analysis_result = {
                "row_labels": selected_row_labels.copy(),
                "model": selected_model,
                "predictions": selected_predictions,
                "metrics": row_metrics,
                "confusion_counts": row_confusion,
            }
        if load_row_clicked and row_to_load is not None:
            selected_prediction = row_labels[row_to_load]
            raw_row = raw_data.iloc[selected_prediction["row_number"] - 2]
            clean_row = add_features(pd.DataFrame([raw_row])).iloc[0]
            field_keys = {
                "title": "job_title",
                "location": "job_location",
                "salary_range": "job_salary",
                "industry": "job_industry",
                "function": "job_function",
                "employment_type": "job_employment_type",
                "required_experience": "job_experience",
                "required_education": "job_education",
                "company_profile": "job_company_profile",
                "description": "job_description",
                "requirements": "job_requirements",
                "benefits": "job_benefits",
                "telecommuting": "job_remote",
                "has_company_logo": "job_logo",
                "has_questions": "job_questions",
            }
            for field, widget_key in field_keys.items():
                value = clean_row[field]
                st.session_state[widget_key] = (
                    bool(value)
                    if field in BINARY_FORM_FIELDS
                    else str(value)
                )
            st.session_state.loaded_csv_row_number = selected_prediction[
                "row_number"
            ]
            st.session_state.csv_analysis_result = None
            st.session_state.analysis_result = None
            st.rerun()

        csv_result = st.session_state.csv_analysis_result
        if csv_result is not None:
            if (
                csv_result["row_labels"] != selected_row_labels
                or csv_result["model"] != selected_model
            ):
                st.info(
                    "The selected rows or model changed. Select **Analyze "
                    "selected CSV rows** to refresh the results."
                )
            else:
                if len(csv_result["predictions"]) == 1:
                    prediction = csv_result["predictions"][0]
                    probability = prediction["fraud_probability"]
                    predicted = prediction["predicted_label"]
                    actual = prediction["actual_label"]
                    result_columns = st.columns(4)
                    result_columns[0].metric("Fraud risk score", f"{probability:.1%}")
                    result_columns[1].metric(
                        "Model prediction",
                        "Fraudulent" if predicted else "Genuine",
                    )
                    result_columns[2].metric(
                        "Known CSV label",
                        "Fraudulent" if actual else "Genuine",
                    )
                    result_columns[3].metric(
                        "Row check",
                        "Correct" if predicted == actual else "Mismatch",
                    )
                    st.caption(
                        f"Model certainty (not calibrated): "
                        f"{max(probability, 1 - probability):.1%} · "
                        f"Selected model: {selected_model} · CSV row "
                        f"{prediction['row_number']}"
                    )
                    with st.expander("View this posting's original CSV fields"):
                        st.dataframe(
                            raw_data.iloc[[prediction["row_number"] - 2]],
                            width="stretch",
                        )
                else:
                    st.markdown(
                        f"#### Metrics for {len(csv_result['predictions'])} selected rows"
                    )
                    result_columns = st.columns(7)
                    for column, metric_name in zip(
                        result_columns,
                        (
                            "Accuracy",
                            "Balanced accuracy",
                            "Precision",
                            "Recall",
                            "F1",
                            "ROC-AUC",
                            "Specificity",
                        ),
                    ):
                        value = csv_result["metrics"][metric_name]
                        column.metric(
                            metric_name,
                            "N/A" if value is None else f"{value:.1%}",
                        )
                    st.caption(
                        "These metrics apply only to your selected rows; small "
                        "selection counts are unstable and are not a dataset-wide "
                        "performance estimate."
                    )
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "CSV row": prediction["row_number"],
                                    "Title": str(
                                        raw_data.iloc[
                                            prediction["row_number"] - 2
                                        ].get("title", "")
                                    ),
                                    "Fraud score": prediction[
                                        "fraud_probability"
                                    ],
                                    "Prediction": (
                                        "Fraudulent"
                                        if prediction["predicted_label"]
                                        else "Genuine"
                                    ),
                                    "Known label": (
                                        "Fraudulent"
                                        if prediction["actual_label"]
                                        else "Genuine"
                                    ),
                                    "Correct": (
                                        prediction["predicted_label"]
                                        == prediction["actual_label"]
                                    ),
                                }
                                for prediction in csv_result["predictions"]
                            ]
                        ).style.format({"Fraud score": "{:.1%}"}),
                        width="stretch",
                    )
                    with st.expander("Confusion counts for selected rows"):
                        st.json(csv_result["confusion_counts"])
    else:
        st.info("No valid labeled CSV rows are available for out-of-fold scoring.")

with st.expander("How the detector works"):
    st.markdown(
        "The app compares Decision Tree, Naive Bayes, Logistic Regression, "
        "SVM, and Random Forest, each with and without chi-square feature "
        "selection. It starts with the configuration that has the highest "
        "validation F1 score (using ROC-AUC to break ties), but you can "
        "select any trained configuration. The model score is based on "
        "training-data patterns; separate text checks point out possible "
        "warning signs and do not change the model score."
    )
    st.caption(
        "The included demo data is fictional and is only for trying the app. "
        "Use representative labeled data to assess real performance. "
        "A prediction is not proof that a posting is genuine or fraudulent."
    )

col1, col2 = st.columns([5, 4])

with col1:
    st.markdown("### 📝 Job details")
    form_widget_defaults = {
        "job_title": "",
        "job_location": "",
        "job_salary": "",
        "job_industry": "",
        "job_function": "",
        "job_employment_type": "",
        "job_experience": "",
        "job_education": "",
        "job_company_profile": "",
        "job_description": "",
        "job_requirements": "",
        "job_benefits": "",
        "job_remote": False,
        "job_logo": False,
        "job_questions": False,
    }
    for key, default in form_widget_defaults.items():
        st.session_state.setdefault(key, default)
    if st.session_state.get("loaded_csv_row_number") is not None:
        st.info(
            f"Loaded fields from CSV row "
            f"{st.session_state.loaded_csv_row_number}. Review or edit them "
            "before analysis."
        )

    title = st.text_input(
        "Job title", placeholder="e.g. Python Developer Intern", key="job_title"
    )
    location = st.text_input(
        "Location", placeholder="e.g. IN, Hyderabad", key="job_location"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        salary = st.text_input(
            "Salary / stipend", placeholder="e.g. 10000-15000", key="job_salary"
        )
    with c2:
        industry = st.text_input(
            "Industry", placeholder="Information Technology", key="job_industry"
        )
    with c3:
        function = st.text_input(
            "Function", placeholder="Engineering", key="job_function"
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        employment_type = st.text_input(
            "Employment type", key="job_employment_type"
        )
    with c5:
        experience = st.text_input("Experience level", key="job_experience")
    with c6:
        education = st.text_input("Education", key="job_education")

    profile = st.text_area(
        "Company profile",
        placeholder="About the company...",
        height=100,
        key="job_company_profile",
    )
    description = st.text_area(
        "Job description",
        placeholder="Paste the job posting here...",
        height=150,
        key="job_description",
    )

    c7, c8 = st.columns(2)
    with c7:
        requirements = st.text_area("Requirements", key="job_requirements")
    with c8:
        benefits = st.text_area("Benefits", key="job_benefits")

    c9, c10, c11 = st.columns(3)
    with c9:
        remote = st.checkbox(
            "Remote job",
            help="Select this if the posting says the work can be done remotely.",
            key="job_remote",
        )
    with c10:
        logo = st.checkbox(
            "Company logo present",
            help="Select this only if the employer's logo appears on the posting.",
            key="job_logo",
        )
    with c11:
        questions = st.checkbox(
            "Screening questions",
            help="Select this if the application includes questions to screen applicants.",
            key="job_questions",
        )

    form = {
        "title": title or "",
        "location": location or "",
        "salary_range": salary or "",
        "industry": industry or "",
        "function": function or "",
        "employment_type": employment_type or "",
        "required_experience": experience or "",
        "required_education": education or "",
        "company_profile": profile or "",
        "description": description or "",
        "requirements": requirements or "",
        "benefits": benefits or "",
        "telecommuting": int(bool(remote)),
        "has_company_logo": int(bool(logo)),
        "has_questions": int(bool(questions)),
    }
    analyze_clicked = st.button(
        "🔍 Analyze posting", type="primary", disabled=assets is None
    )

with col2:
    if analyze_clicked and not any(
        str(form[field]).strip()
        for field in ("title", "description", "requirements", "company_profile")
    ):
        st.session_state.analysis_result = None
        st.error(
            "Enter a job title, description, requirements, or company profile "
            "before analyzing. An empty form cannot produce a useful score."
        )
    elif analyze_clicked:
        all_text = " ".join(str(v) for v in form.values())
        warning_signals = []
        missing_info = []

        if not form["company_profile"].strip():
            missing_info.append("Company profile was not provided")
        if not form["has_company_logo"]:
            missing_info.append("Company logo was not indicated")
        if re.search(r"@(gmail|yahoo|outlook|hotmail)\.", all_text, re.I):
            warning_signals.append("A free email address appears in the posting")
        if re.search(r"registration fee|deposit|western union|whatsapp|pay ", all_text, re.I):
            warning_signals.append("The posting mentions payment or messaging apps")
        if re.search(r"no experience|urgent|earn \w+ (daily|weekly)|guaranteed", all_text, re.I):
            warning_signals.append("The posting uses urgent or unusually attractive wording")

        row_df = add_features(pd.DataFrame([form]))
        probability = float(
            assets["models"][selected_model].predict_proba(row_df)[0, 1]
        )

        if probability >= 0.5:
            color = "#dc2626"
            verdict = "⚠️ Elevated model fraud score"
            risk = "The trained model found a stronger match to fraudulent examples."
        elif probability >= 0.3:
            color = "#d97706"
            verdict = "🤔 Mixed model signal"
            risk = "The model result is uncertain; review the posting carefully."
        elif warning_signals:
            color = "#d97706"
            verdict = "🔎 Review before proceeding"
            risk = "The model score is low, but separate warning signs were found."
        else:
            color = "#059669"
            verdict = "✅ No strong fraud signal detected"
            risk = "A low model score is not proof that this posting is genuine."

        st.session_state.analysis_result = {
            "form": form.copy(),
            "model": selected_model,
            "probability": probability,
            "warning_signals": warning_signals,
            "missing_info": missing_info,
            "verdict": verdict,
            "risk": risk,
            "color": color,
        }

    result = st.session_state.analysis_result
    if result is not None and assets is not None:
        if result["form"] != form or result["model"] != selected_model:
            st.info(
                "The form or selected model changed since this result was "
                "calculated. Select **Analyze posting** to refresh it."
            )
        percentage = round(result["probability"] * 100, 1)
        signals_html = "".join(
            f'<div class="fj-flag">🚩 {_html.escape(signal)}</div>'
            for signal in result["warning_signals"]
        )
        if not signals_html:
            signals_html = '<div class="fj-ok">No rule-based warning signs detected</div>'
        missing_html = "".join(
            f'<div class="fj-info">ℹ️ {_html.escape(item)}</div>'
            for item in result["missing_info"]
        )
        missing_section = (
            f"<h3>Missing information</h3>{missing_html}" if missing_html else ""
        )
        signal_count = len(result["warning_signals"])
        missing_count = len(result["missing_info"])
        signal_summary = (
            f"{signal_count} warning signal{'s' if signal_count != 1 else ''}"
            f" · {missing_count} missing detail{'s' if missing_count != 1 else ''}"
        )
        st.markdown(f"""
        <div class="fj-card">
          <div class="fj-top">
            <div class="fj-gauge" style="--p:{percentage};--c:{result['color']}">
              <div class="fj-gauge-inner">
                <b>{percentage}%</b>
                <small>model fraud score</small>
              </div>
            </div>
            <div>
              <span class="fj-badge" style="background:{result['color']}">{_html.escape(result['verdict'])}</span>
              <p class="fj-risk-text">
                {_html.escape(result['risk'])} Verify the employer through an independent official source. Never pay to apply.
              </p>
              <p class="fj-tip">Analyzed with {_html.escape(result['model'])}</p>
            </div>
          </div>
          <p class="fj-tip">
            This manual score comes from the final model fitted on all labeled
            CSV rows. If you loaded a CSV row, use its out-of-fold score above
            for an independent check; the final model has seen the training rows.
          </p>
          <h3>Possible warning signs</h3>
          <p class="fj-tip">The model score is not a guaranteed real-world probability. These text checks are separate from it; missing information is not proof of fraud.</p>
          <p class="fj-signal-summary">Independent checks: {_html.escape(signal_summary)}</p>
          {signals_html}
          {missing_section}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="fj-card" style="padding:28px">
          <h3 style="margin-top:0">Your result will appear here</h3>
          <p>To check a posting:</p>
          <ol>
            <li>Train the model from the sidebar using the demo or your labeled CSV.</li>
            <li>Enter the details you have; the title and description are most useful.</li>
            <li>Select <strong>Analyze posting</strong> to see the model estimate and warning signs.</li>
          </ol>
          <p class="fj-tip">A missing detail is fine—leave it blank rather than guessing.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<p class="fj-foot">
  Educational screening aid only: a model estimate is not proof that a job is
  genuine or fraudulent. Verify employers independently and never pay to apply.
</p>
""", unsafe_allow_html=True)
