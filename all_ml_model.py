import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# ML Imports
# ==========================

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

# ==========================
# STEP 1 : LOAD DATASET
# ==========================

df = pd.read_csv(
    "/storage/emulated/0/Datasets/heart.csv"
)

print("="*50)
print("DATASET SHAPE")
print("="*50)
print(df.shape)

print("\nINFO")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nTARGET DISTRIBUTION")
print(df["target"].value_counts())

# ==========================
# CORRELATION HEATMAP
# ==========================

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(),
    cmap="coolwarm"
)

plt.title(
    "Correlation Heatmap"
)

plt.show()

# ==========================
# STEP 2 : PREPROCESSING
# ==========================

X = df.drop(
    "target",
    axis=1
)

y = df["target"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_s = scaler.fit_transform(
    X_train
)

X_test_s = scaler.transform(
    X_test
)

# ==========================
# STEP 3 : MODELS
# ==========================

models = {

    "Logistic Regression":
    LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree":
    DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    ),

    "Random Forest":
    RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=42
    ),

    "SVM":
    SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),

    "KNN":
    KNeighborsClassifier(
        n_neighbors=5
    ),

    "Naive Bayes":
    GaussianNB()
}

# ==========================
# STEP 4 : TRAIN + EVALUATE
# ==========================

results = []

plt.figure(figsize=(10,8))

for name,model in models.items():

    model.fit(
        X_train_s,
        y_train
    )

    y_pred = model.predict(
        X_test_s
    )

    y_prob = model.predict_proba(
        X_test_s
    )[:,1]

    acc = accuracy_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    cv = cross_val_score(
        model,
        X_train_s,
        y_train,
        cv=5,
        scoring="accuracy"
    ).mean()

    results.append(
        [
            name,
            acc,
            f1,
            auc,
            cv
        ]
    )

    print("\n")
    print("="*60)
    print(name)
    print("="*60)

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # ROC Curve

    fpr,tpr,_ = roc_curve(
        y_test,
        y_prob
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.2f})"
    )

# ==========================
# ROC PLOT
# ==========================

plt.plot(
    [0,1],
    [0,1],
    '--'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.grid(True)

plt.show()

# ==========================
# STEP 5 : RESULT TABLE
# ==========================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "F1",
        "ROC_AUC",
        "CV_Score"
    ]
)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n")
print("="*60)
print("FINAL RESULTS")
print("="*60)

print(results_df)

# ==========================
# MODEL COMPARISON BARPLOT
# ==========================

plt.figure(figsize=(12,8))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.xticks(rotation=45)

plt.ylabel("Accuracy")

plt.title(
    "Model Accuracy Comparison"
)

plt.tight_layout()

plt.show()

# ==========================
# RANDOM FOREST
# FEATURE IMPORTANCE
# ==========================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

importance = rf.feature_importances_

indices = np.argsort(
    importance
)[::-1]

plt.figure(figsize=(12,8))

plt.barh(
    [X.columns[i] for i in indices],
    importance[indices]
)

plt.title(
    "Random Forest Feature Importance"
)

plt.xlabel(
    "Importance"
)

plt.tight_layout()

plt.show()

# ==========================
# STEP 6 : BEST MODEL
# ==========================

best_model = results_df.iloc[0]

print("\n")
print("="*60)
print("BEST MODEL")
print("="*60)

print(best_model)

print("\nConclusion:")

print(
    f"""
Best Model : {best_model['Model']}

Accuracy   : {best_model['Accuracy']:.4f}

F1 Score   : {best_model['F1']:.4f}

ROC-AUC    : {best_model['ROC_AUC']:.4f}

CV Score   : {best_model['CV_Score']:.4f}
"""
)