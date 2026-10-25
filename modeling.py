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
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
MODEL_ASSET_VERSION = 5
CAT_COLS = [
    "location",
    "salary_range",
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
ALL_INPUTS = TEXT_COLS + CAT_COLS + BINARY_COLS
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

    for column in TEXT_COLS + CAT_COLS:
        data[column] = data[column].fillna("").astype(str)

    raw_text = data[TEXT_COLS].agg(" ".join, axis=1)
    for column in TEXT_COLS:
        data[column] = data[column].map(clean_text)
    for column in CAT_COLS:
        data[column] = data[column].str.strip().str.lower().replace("", "unknown")

    for column in BINARY_COLS:
        source = data[column]
        normalized = source.astype(str).str.strip().str.lower()
        mapped = normalized.map(
            {"1": 1, "true": 1, "yes": 1, "y": 1, "0": 0, "false": 0, "no": 0, "n": 0}
        )
        numeric = pd.to_numeric(source, errors="coerce")
        data[column] = mapped.where(
            mapped.notna(), numeric.where(numeric.isin([0, 1]), 0)
        ).fillna(0).astype(int)

    data["text"] = (
        data[TEXT_COLS].agg(" ".join, axis=1).str.strip().replace("", "emptytext")
    )
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
                    min_df=1,
                ),
                "text",
            ),
            ("cat", _make_encoder(), CAT_COLS),
            ("num", MinMaxScaler(), NUM_COLS),
        ],
        remainder="drop",
    )


def _models(calibration_cv=3):
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=30, class_weight="balanced", random_state=42
        ),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "SVM": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced"), cv=calibration_cv
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }


def calculate_metrics(y_true, predictions, probabilities):
    """Return binary classification metrics for already-computed predictions."""
    has_both_classes = len(set(y_true)) == 2
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
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
    }, {
        "True negative": int(tn),
        "False positive": int(fp),
        "False negative": int(fn),
        "True positive": int(tp),
    }


def train_best_model(raw_dataframe):
    """Train and evaluate each notebook model configuration."""
    if "fraudulent" not in raw_dataframe.columns:
        raise ValueError("The CSV must include a 'fraudulent' target column.")

    data = raw_dataframe.copy()
    data["_source_row_number"] = range(2, len(data) + 2)
    data["fraudulent"] = pd.to_numeric(data["fraudulent"], errors="coerce")
    data = data.dropna(subset=["fraudulent"]).copy()
    if not set(data["fraudulent"].unique()).issubset({0, 1}):
        raise ValueError(
            "The 'fraudulent' column may only contain 0 (genuine) or 1 "
            "(fraudulent) labels."
        )
    data["fraudulent"] = data["fraudulent"].astype(int)
    if set(data["fraudulent"].unique()) != {0, 1}:
        raise ValueError("The 'fraudulent' column must contain both 0 and 1.")
    if data["fraudulent"].value_counts().min() < 3:
        raise ValueError("At least three examples of each class are required.")

    featured = add_features(data)
    X = featured.drop(columns=["fraudulent", "_source_row_number"])
    y = featured["fraudulent"]
    folds = min(5, int(y.value_counts().min()))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    split_indices = list(splitter.split(X, y))
    estimators = _models(calibration_cv=2)
    candidates = [
        {
            "name": name,
            "feature_selection": feature_selection,
            "label": (
                f"{name} · "
                f"{'with' if feature_selection else 'without'} "
                "feature selection"
            ),
            "probabilities": [None] * len(y),
            "predictions": [None] * len(y),
            "folds": folds,
        }
        for feature_selection in (False, True)
        for name in estimators
    ]
    candidate_lookup = {
        (candidate["name"], candidate["feature_selection"]): candidate
        for candidate in candidates
    }
    for train_indices, validation_indices in split_indices:
        X_train = X.iloc[train_indices]
        X_validation = X.iloc[validation_indices]
        y_train = y.iloc[train_indices]
        preprocessor = _make_preprocessor()
        X_train_prepared = preprocessor.fit_transform(X_train, y_train)
        X_validation_prepared = preprocessor.transform(X_validation)
        train_views = {False: X_train_prepared}
        validation_views = {False: X_validation_prepared}
        selector = SelectKBest(
            score_func=chi2, k=min(2000, X_train_prepared.shape[1])
        )
        train_views[True] = selector.fit_transform(X_train_prepared, y_train)
        validation_views[True] = selector.transform(X_validation_prepared)

        for feature_selection in (False, True):
            for name, estimator in estimators.items():
                fold_classifier = clone(estimator)
                fold_classifier.fit(train_views[feature_selection], y_train)
                fold_probabilities = fold_classifier.predict_proba(
                    validation_views[feature_selection]
                )[:, 1]
                fold_predictions = fold_classifier.predict(
                    validation_views[feature_selection]
                )
                candidate = candidate_lookup[(name, feature_selection)]
                for position, probability, prediction in zip(
                    validation_indices, fold_probabilities, fold_predictions
                ):
                    candidate["probabilities"][position] = float(probability)
                    candidate["predictions"][position] = int(prediction)

    for candidate in candidates:
        probabilities = candidate.pop("probabilities")
        predictions = candidate.pop("predictions")
        candidate["metrics"], candidate["confusion_counts"] = calculate_metrics(
            y, predictions, probabilities
        )
        candidate["validation_predictions"] = [
            {
                "row_number": int(source_row_number),
                "actual_label": int(actual),
                "predicted_label": int(predicted),
                "fraud_probability": float(probability),
            }
            for source_row_number, actual, predicted, probability in zip(
                data["_source_row_number"], y, predictions, probabilities
            )
        ]

    best = max(
        candidates,
        key=lambda candidate: (
            candidate["metrics"]["F1"],
            candidate["metrics"]["ROC-AUC"],
        ),
    )

    # Fit a separate production-ready model for every selectable configuration.
    full_preprocessor = _make_preprocessor()
    full_prepared = full_preprocessor.fit_transform(X, y)
    full_selectors = {
        False: None,
        True: SelectKBest(
            score_func=chi2, k=min(2000, full_prepared.shape[1])
        ).fit(full_prepared, y),
    }
    trained_models = {}
    for candidate in candidates:
        selector = full_selectors[candidate["feature_selection"]]
        model_features = (
            selector.transform(full_prepared)
            if selector is not None
            else full_prepared
        )
        classifier = clone(estimators[candidate["name"]])
        classifier.fit(model_features, y)
        steps = [("prep", full_preprocessor)]
        if selector is not None:
            steps.append(("feature_selection", selector))
        steps.append(("classifier", classifier))
        trained_models[candidate["label"]] = Pipeline(steps)

    return {
        "asset_version": MODEL_ASSET_VERSION,
        "models": trained_models,
        "model_scores": {
            candidate["label"]: candidate["metrics"] for candidate in candidates
        },
        "validation_predictions": {
            candidate["label"]: candidate["validation_predictions"]
            for candidate in candidates
        },
        "confusion_counts": {
            candidate["label"]: candidate["confusion_counts"]
            for candidate in candidates
        },
        **ASSET_FIELDS,
        "metrics": best["metrics"],
        "selected_model": best["label"],
        "feature_selection": best["feature_selection"],
        "training_rows": len(data),
        "validation_rows": len(y),
        "validation_folds": folds,
    }
