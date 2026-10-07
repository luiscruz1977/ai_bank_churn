import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


# 1. Load data
df = pd.read_csv("data/customer_churn.csv")

X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]


# 2. Feature types
categorical_features = ["gender"]

numeric_features = [
    "age",
    "tenure_months",
    "num_products",
    "balance",
    "credit_score",
    "is_active_member",
    "estimated_salary",
    "has_credit_card",
    "num_transactions_last_month",
    "support_tickets_last_6m"
]


# 3. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ]
)


# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 5. Models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
}


# 6. Train and evaluate
results = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    results[name] = {
        "pipeline": pipeline,
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities)
    }

    print(f"\n{name}")
    print(f"Precision: {results[name]['precision']:.3f}")
    print(f"Recall:    {results[name]['recall']:.3f}")
    print(f"F1:        {results[name]['f1']:.3f}")
    print(f"ROC-AUC:   {results[name]['roc_auc']:.3f}")


# 7. Select model by ROC-AUC
best_name = max(
    results,
    key=lambda name: results[name]["roc_auc"]
)

best_model = results[best_name]["pipeline"]

print(f"\nBest model: {best_name}")


# 8. Save model
joblib.dump(best_model, "ml/model.pkl")

print("Model saved to ml/model.pkl")