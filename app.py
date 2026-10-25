import re
from pathlib import Path

import streamlit as st
import pandas as pd
import html as _html
from modeling import MODEL_ASSET_VERSION, add_features, train_best_model

DEMO_DATA_PATH = Path(__file__).with_name("demo_jobs.csv")

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

if (
    st.session_state.model_assets is not None
    and st.session_state.model_assets.get("asset_version") != MODEL_ASSET_VERSION
):
    st.session_state.model_assets = None
    st.session_state.analysis_result = None
    st.session_state.csv_analysis_result = None
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
            with st.spinner("Training and comparing the notebook's 10 model configurations..."):
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
        f"best validation F1 {assets['metrics']['F1']:.3f}"
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
    st.markdown("#### Selected model's validation scores")
    metric_columns = st.columns(5)
    for column, metric_name in zip(
        metric_columns,
        ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC"),
    ):
        column.metric(metric_name, f"{selected_metrics[metric_name]:.1%}")
    st.caption(
        f"Selected model: {selected_model} · "
        f"training rows: {assets['training_rows']:,} · "
        f"validation rows: {assets['validation_rows']:,} · "
        f"source: {assets['training_source']}. Scores are measured on "
        "held-out rows and do not guarantee performance on new postings."
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
        "The selected model is trained on the uploaded data. Validation "
        "scores are for comparing these configurations, not a guarantee of "
        "future performance."
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
            "This dataset is small, so validation scores can change a lot "
            "with different postings. The demo is synthetic and is not a "
            "real-world accuracy test; use a larger, representative labeled dataset."
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
    st.subheader("Score one held-out CSV posting")
    st.caption(
        "Choose a row the model did not train on. Its known label is shown "
        "beside the selected model's prediction so you can inspect a real "
        "held-out example without scoring a row the model memorized."
    )
    validation_rows = assets["validation_predictions"][selected_model]
    raw_data = st.session_state.training_data_raw
    row_labels = {}
    for prediction in validation_rows:
        source_row = prediction["row_number"] - 2
        raw_row = raw_data.iloc[source_row]
        title_value = str(raw_row.get("title", "")).strip() or "Untitled posting"
        row_labels[
            f"CSV row {prediction['row_number']} · {title_value[:90]}"
        ] = prediction

    if row_labels:
        selected_row_label = st.selectbox(
            "Held-out posting",
            list(row_labels),
            key="validation_row_choice",
        )
        score_row_clicked = st.button("Score selected CSV row")
        if score_row_clicked:
            selected_prediction = row_labels[selected_row_label]
            st.session_state.csv_analysis_result = {
                "row_label": selected_row_label,
                "model": selected_model,
                **selected_prediction,
            }

        csv_result = st.session_state.csv_analysis_result
        if csv_result is not None:
            if (
                csv_result["row_label"] != selected_row_label
                or csv_result["model"] != selected_model
            ):
                st.info(
                    "The selected row or model changed. Select **Score selected "
                    "CSV row** to refresh this result."
                )
            else:
                probability = csv_result["fraud_probability"]
                predicted = csv_result["predicted_label"]
                actual = csv_result["actual_label"]
                result_columns = st.columns(4)
                result_columns[0].metric("Fraud risk score", f"{probability:.1%}")
                result_columns[1].metric(
                    "Model prediction", "Fraudulent" if predicted else "Genuine"
                )
                result_columns[2].metric(
                    "Known CSV label", "Fraudulent" if actual else "Genuine"
                )
                result_columns[3].metric(
                    "Held-out check", "Correct" if predicted == actual else "Mismatch"
                )
                st.caption(
                    f"Model confidence: {max(probability, 1 - probability):.1%} · "
                    f"Selected model: {selected_model} · CSV row "
                    f"{csv_result['row_number']}"
                )
                with st.expander("View this held-out posting's original CSV fields"):
                    st.dataframe(
                        raw_data.iloc[[csv_result["row_number"] - 2]],
                        width="stretch",
                    )
    else:
        st.info("No held-out rows are available for this trained dataset.")

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
    title = st.text_input("Job title", placeholder="e.g. Python Developer Intern")
    location = st.text_input("Location", placeholder="e.g. IN, Hyderabad")

    c1, c2, c3 = st.columns(3)
    with c1:
        salary = st.text_input("Salary / stipend", placeholder="e.g. 10000-15000")
    with c2:
        industry = st.text_input("Industry", placeholder="Information Technology")
    with c3:
        function = st.text_input("Function", placeholder="Engineering")

    c4, c5, c6 = st.columns(3)
    with c4:
        employment_type = st.selectbox("Employment type", ["", "Full-time", "Part-time", "Contract", "Temporary", "Other"])
    with c5:
        experience = st.selectbox("Experience level", ["", "Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Not Applicable"])
    with c6:
        education = st.selectbox("Education", ["", "Bachelor's Degree", "Master's Degree", "High School or equivalent", "Certification", "Unspecified"])

    profile = st.text_area("Company profile", placeholder="About the company...", height=100)
    description = st.text_area("Job description", placeholder="Paste the job posting here...", height=150)

    c7, c8 = st.columns(2)
    with c7:
        requirements = st.text_area("Requirements")
    with c8:
        benefits = st.text_area("Benefits")

    c9, c10, c11 = st.columns(3)
    with c9:
        remote = st.checkbox(
            "Remote job",
            help="Select this if the posting says the work can be done remotely.",
        )
    with c10:
        logo = st.checkbox(
            "Company logo present",
            help="Select this only if the employer's logo appears on the posting.",
        )
    with c11:
        questions = st.checkbox(
            "Screening questions",
            help="Select this if the application includes questions to screen applicants.",
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
