# ==============================================================================
# CELL 2 - Imports
# ==============================================================================

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    ConfusionMatrixDisplay
)

# ==============================================================================
# CELL 3 - Load & split data
# ==============================================================================

data = load_breast_cancer()
X, y = data.data, data.target

# stratify=y ensures the same malignant/benign ratio in train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train: {X_train.shape} Test: {X_test.shape}")
# Output: Train: (455, 30) Test: (114, 30)

# ==============================================================================
# CELL 4 - Configure MLFlow tracking URI and experiment
# ==============================================================================

# SQLite keeps everything in a local file - great for Colab demos.
# In Azure ML, replace this with your workspace tracking URI.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# All runs logged below will appear under this experiment tab in the UI
mlflow.set_experiment("breast-cancer-classifier")

# ==============================================================================
# CELL 5 - RUN A: Manual logging
# ==============================================================================
# We log everything explicitly so we understand
# exactly what autolog() does later.

MAX_DEPTH    = 4
MIN_SAMPLES  = 5
CRITERION    = "gini"

with mlflow.start_run(run_name="manual-logging") as run:
    # — Step 1: Log every hyperparameter we pass to the model —
    mlflow.log_param("max_depth",          MAX_DEPTH)
    mlflow.log_param("min_samples_split",  MIN_SAMPLES)
    mlflow.log_param("criterion",          CRITERION)
    mlflow.log_param("random_state",       42)
    
    # — Step 2: Train —
    clf = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES,
        criterion=CRITERION,
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # — Step 3: Log evaluation metrics —
    mlflow.log_metric("accuracy",  accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall",    recall_score(y_test, y_pred))
    mlflow.log_metric("f1",        f1_score(y_test, y_pred))
    
    # — Step 4: Save confusion matrix as a PNG artefact —
    # Artefacts are files (images, CSVs, plots) attached to a run
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png") # uploads file to run
    plt.close()
    
    # — Step 5: Log the serialised model object itself —
    mlflow.sklearn.log_model(clf, "decision-tree-model")
    
    print(f"Run ID : {run.info.run_id}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# ==============================================================================
# CELL 6 - RUN B: Autolog (same result, zero manual code)
# ==============================================================================
# mlflow.sklearn.autolog() monkey-patches fit() so that the
# moment you call clf.fit() it automatically captures:
#  - all constructor parameters
#  - training metrics
#  - the fitted model
#  - an input example (sample of X_train)
#  - a model signature (inferred input/output schema)

mlflow.sklearn.autolog(
    log_input_examples=True,   # attaches a sample of X_train
    log_model_signatures=True, # infers & stores input/output schema
    log_models=True            # saves the fitted model object
)

with mlflow.start_run(run_name="autolog-run"):
    clf2 = DecisionTreeClassifier(max_depth=6, random_state=42)
    clf2.fit(X_train, y_train)
    # autolog fires at fit() - nothing else needed!
    print("autolog run complete - check the UI for logged params & metrics")

# ==============================================================================
# CELL 7 - Launch the MLFlow Tracking UI in Colab
# ==============================================================================
# This starts the server in the background on port 5000.
# In Colab, use the port-forwarding panel (or install pyngrok).
# !mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 &
print("UI running at: http://localhost:5000")
# Navigate to the experiment, select both runs, click Compare