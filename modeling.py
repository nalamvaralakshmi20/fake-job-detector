import re

import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
CAT_COLS = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]
BINARY_COLS = ["telecommuting", "has_company_logo", "has_questions"]
NUM_COLS = BINARY_COLS + [
    "text_len",
    "caps_ratio",
    "has_salary",
    "has_profile",
    "n_links",
    "n_exclaim",
]
ALL_INPUTS = TEXT_COLS + CAT_COLS + ["location", "salary_range"] + BINARY_COLS
ASSET_FIELDS = {
    "TEXT_COLS": TEXT_COLS,
    "CAT_COLS": CAT_COLS,
    "BINARY_COLS": BINARY_COLS,
    "NUM_COLS": NUM_COLS,
    "ALL_INPUTS": ALL_INPUTS,
}


def clean_text(value):
    value = str(value).lower()
    value = re.sub(r"http\S+|www\.\S+", " url ", value)
    value = re.sub(r"\S+@\S+", " email ", value)
    value = re.sub(r"[^a-z\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def add_features(dataframe):
    data = dataframe.copy()
    for column in ALL_INPUTS:
        if column not in data.columns:
            data[column] = 0 if column in BINARY_COLS else ""

    for column in TEXT_COLS + CAT_COLS + ["location", "salary_range"]:
        data[column] = data[column].fillna("").astype(str)

    for column in BINARY_COLS:
        data[column] = (
            pd.to_numeric(data[column], errors="coerce").fillna(0).astype(int)
        )

    for column in CAT_COLS:
        data[column] = data[column].replace("", "unknown")

    raw_text = data[TEXT_COLS].agg(" ".join, axis=1)
    data["text"] = raw_text.map(clean_text)
    data["text_len"] = raw_text.str.len()
    data["caps_ratio"] = raw_text.map(
        lambda value: sum(char.isupper() for char in value) / max(len(value), 1)
    )
    data["has_salary"] = (data["salary_range"].str.strip() != "").astype(int)
    data["has_profile"] = (data["company_profile"].str.strip() != "").astype(int)
    data["n_links"] = raw_text.str.count(r"http|www\.|@")
    data["n_exclaim"] = raw_text.str.count("!")
    return data


def _make_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    max_features=5000,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                ),
                "text",
            ),
            ("cat", _make_encoder(), CAT_COLS),
            ("num", MinMaxScaler(), NUM_COLS),
        ],
        remainder="drop",
    )


def _models():
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=30, class_weight="balanced", random_state=42
        ),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "SVM": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced"), cv=3
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }


def train_best_model(raw_dataframe):
    """Train and select a model using the notebook's F1/ROC-AUC workflow."""
    if "fraudulent" not in raw_dataframe.columns:
        raise ValueError("The CSV must include a 'fraudulent' target column.")

    data = raw_dataframe.copy()
    data["fraudulent"] = pd.to_numeric(data["fraudulent"], errors="coerce")
    data = data.dropna(subset=["fraudulent"]).copy()
    data["fraudulent"] = data["fraudulent"].astype(int)
    if set(data["fraudulent"].unique()) != {0, 1}:
        raise ValueError("The 'fraudulent' column must contain both 0 and 1.")
    if data["fraudulent"].value_counts().min() < 3:
        raise ValueError("At least three examples of each class are required.")

    featured = add_features(data)
    X = featured.drop(columns=["fraudulent"])
    y = featured["fraudulent"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    candidates = []
    for feature_selection in (False, True):
        for name, estimator in _models().items():
            preprocessor = _make_preprocessor()
            X_train_prepared = preprocessor.fit_transform(X_train, y_train)
            selector = None
            X_train_model = X_train_prepared
            if feature_selection:
                selector = SelectKBest(
                    score_func=chi2, k=min(2000, X_train_prepared.shape[1])
                )
                X_train_model = selector.fit_transform(X_train_prepared, y_train)

            classifier = clone(estimator)
            classifier.fit(X_train_model, y_train)
            X_test_prepared = preprocessor.transform(X_test)
            X_test_model = (
                selector.transform(X_test_prepared)
                if selector is not None
                else X_test_prepared
            )
            probabilities = classifier.predict_proba(X_test_model)[:, 1]
            predictions = classifier.predict(X_test_model)
            metrics = {
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1": f1_score(y_test, predictions, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, probabilities),
            }
            candidates.append(
                {
                    "name": name,
                    "feature_selection": feature_selection,
                    "metrics": metrics,
                }
            )

    best = max(
        candidates,
        key=lambda candidate: (
            candidate["metrics"]["F1"],
            candidate["metrics"]["ROC-AUC"],
        ),
    )

    # Refit the selected configuration on all supplied labeled rows.
    preprocessor = _make_preprocessor()
    X_prepared = preprocessor.fit_transform(X, y)
    selector = None
    if best["feature_selection"]:
        selector = SelectKBest(
            score_func=chi2, k=min(2000, X_prepared.shape[1])
        )
        X_prepared = selector.fit_transform(X_prepared, y)
    classifier = clone(_models()[best["name"]])
    classifier.fit(X_prepared, y)
    steps = [("prep", preprocessor)]
    if selector is not None:
        steps.append(("feature_selection", selector))
    steps.append(("classifier", classifier))

    return {
        "model": Pipeline(steps),
        **ASSET_FIELDS,
        "metrics": best["metrics"],
        "selected_model": best["name"],
        "feature_selection": best["feature_selection"],
        "training_rows": len(data),
    }
