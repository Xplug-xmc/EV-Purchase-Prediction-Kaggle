#IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

#LOAD THE DATA
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(f"\n TOTAL NUMBER OF ROWS AND COLUMN")
print("-------------------------------------")
print(train.shape)
print(test.shape)


#INSPECT THE DATA
print(f"\n FIRST FIVE ROWS AND COLUMN")
print("------------------------------------")
print(train.head())

print(f"\n DATA INFORMTIONS")
print("-------------------------")
print(train.info())

print(f"\n DATA STATS SUMMARY")
print("------------------------")
print(train.describe())

print(f"\n CHECK FOR MISSING VALUES")
print("---------------------------------")
print(train.isnull().sum())


#VISUALIZE THE DATASET
#VISUALIZE THE TARGET
plt.figure(figsize=(6, 4))
sns.countplot(data=train, x="Will_Buy_EV")

plt.title("EV Purchase Distribution")
plt.xlabel("Will Buy EV")
plt.ylabel("Number of Customers")
plt.show()

#VISUALIZE ANNUAL INCOME
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=train,
    x="Will_Buy_EV",
    y="Annual_Income_USD"
)

plt.title("Annual Income vs EV Purchase")
plt.xlabel("Will Buy EV")
plt.ylabel("Annual Income (USD)")

plt.show()

#VISUALIZE ENVIRONMENTAL CONCERN
plt.figure(figsize=(8, 5))

sns.countplot(
    data=train,
    x="Environmental_Concern_Level",
    hue="Will_Buy_EV"
)

plt.title("Environmental Concern vs EV Purchase")
plt.xlabel("Environmental Concern Level")
plt.ylabel("Number of Customers")

plt.show()

#VISUALIZE SUBSIDY AVAILABLE
plt.figure(figsize=(7, 5))

sns.countplot(
    data=train,
    x="Subsidy_Available",
    hue="Will_Buy_EV"
)

plt.title("Subsidy Availability vs EV Purchase")
plt.xlabel("Subsidy Available")
plt.ylabel("Number of Customers")

plt.show()

#VISUALIZE RANGE ANXIETY
plt.figure(figsize=(8, 5))

sns.countplot(
    data=train,
    x="Range_Anxiety_Level",
    hue="Will_Buy_EV"
)

plt.title("Range Anxiety vs EV Purchase")
plt.xlabel("Range Anxiety Level")
plt.ylabel("Number of Customers")

plt.show()

#VISUALIZE HOME CHARGING
plt.figure(figsize=(7, 5))

sns.countplot(
    data=train,
    x="Home_Charging_Possible",
    hue="Will_Buy_EV"
)

plt.title("Home Charging vs EV Purchase")
plt.xlabel("Home Charging Possible")
plt.ylabel("Number of Customers")

plt.show()


#PREPARE X AND Y
X = train.drop(columns = ["Will_Buy_EV", "id"])
Y = train["Will_Buy_EV"]

print(f"\n CHECK Y AND Y")
print("--------------------")
print(X.shape)
print(Y.shape)


#SEPARATE NUMERICAL AND CATEGORICAL FEATURES
numeric_features = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level"
]

categorical_features = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level"
]


#TRAIN / TEST SPLIT

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)


#CHECK THE SPLIT
print(f"\n CHECK THE SPLIT")
print("-----------------------")
print(f"\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print(f"\nTraining target:")
print(Y_train.value_counts(normalize=True))

print("\n Testing target:")
print(Y_test.value_counts(normalize=True))


#CONVERT THE TARGET TO 0 | 1
y = train["Will_Buy_EV"].map({
    "No": 0,
    "Yes": 1
})

print(f"\n FIRST FIVE TARGET ROWS")
print("----------------------------------")
print(y.head())


# PREPARE THE DATA FOR LOGISTIC REGRESSION USING ONE-HOT ENCODING
preprocessor = ColumnTransformer(transformers=[(
    "num",
    StandardScaler(),
    numeric_features ),

(
    "cat",
    OneHotEncoder(handle_unknown= "ignore"),
    categorical_features
)])


#CREATE THE LOGISTIC REGRESSION PIPELINE
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression( max_iter= 1000))
])


# TRAIN THE MODEL
model.fit(X_train, Y_train)

# MAKE PROBABILITY PREDICTION
test_predictions = model.predict_proba(X_test)[:, 1]

# EVALUATE USING ROC-AUC
baseline_auc = roc_auc_score(
    Y_test,
    test_predictions
)

print(f"\n Baseline ROC-AUC: {baseline_auc:.5f}")

print("\n")


# BUILD A CATBOOST MODEL
print(f"CATBOOST INTERATIONS = 500 EXPERIMENT")
print("-----------------------------------------")

cat_model = CatBoostClassifier(
    iterations= 500,
    learning_rate= 0.05,
    depth= 6,
    loss_function= "Logloss",
    eval_metric= "AUC",
    random_seed= 42,
    verbose= 100
)


# BUILD THE MODEL
cat_model.fit(
    X_train,
    Y_train,
    cat_features= categorical_features,
    eval_set= (X_test, Y_test),
    early_stopping_rounds= 50
)


# MAKE PROBABILITY PREDICTIONS
cat_predictions = cat_model.predict_proba(X_test)[:, 1]


# CALCULATE ROC-AUC
cat_auc = roc_auc_score(
    Y_test,
    cat_predictions
)

print(f"\n CatBoost ROC-AUC: {cat_auc:.5f}")


# COMPARE THE MODELS
print(f"\n MODEL COMPARISON")
print("--------------------")
print(f"Logistic Regression: {baseline_auc:.5f}")
print(f"CatBoost:            {cat_auc:.5f}")


# INSPECT THE IMPORTANT FEATURES USING CATBOOST
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": cat_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)


print(f"\n Feature Importance")
print("--------------------------")
print(feature_importance)


# INVESTIGATE THE STRONGEST FEATURES
# Purchase rate by subsidy
subsidy_rate = train.groupby("Subsidy_Available")["Will_Buy_EV"].apply(
    lambda x: (x == "Yes").mean()
)

print(f"\n EV Purchase Rate by Subsidy:")
print(subsidy_rate)


# Purchase rate by range anxiety
range_rate = train.groupby("Range_Anxiety_Level")["Will_Buy_EV"].apply(
    lambda x: (x == "Yes").mean()
)

print(f"\nEV Purchase Rate by Range Anxiety:")
print(range_rate)


# Purchase rate by environmental concern
environment_rate = train.groupby(
    "Environmental_Concern_Level"
)["Will_Buy_EV"].apply(
    lambda x: (x == "Yes").mean()
)

print(f"\nEV Purchase Rate by Environmental Concern:")
print(environment_rate)


# Purchase rate by home charging
charging_rate = train.groupby(
    "Home_Charging_Possible"
)["Will_Buy_EV"].apply(
    lambda x: (x == "Yes").mean()
)

print(f"\nEV Purchase Rate by Home Charging:")
print(charging_rate)


# INVESTIGATE COMBINATIONS
combo_rate = train.groupby(
    ["Subsidy_Available", "Environmental_Concern_Level"]
)["Will_Buy_EV"].apply(
    lambda x: (x == "Yes").mean()
)


print(f"\n Combination Rate")
print("-----------------------")
print(combo_rate)


print("\n")

#IMPROVE CATBOOST EXPERIMENTS
print(f"CATBOOST INTERATIONS = 1000 EXPERIMENT")
print("---------------------------------------------")

cat_model_1000 = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=200
)

# BUILD THE MODEL
cat_model_1000.fit(
    X_train,
    Y_train,
    cat_features=categorical_features,
    eval_set=(X_test, Y_test),
    early_stopping_rounds=100
)

# MAKE PROBABILITY PREDICTIONS
predictions_1000 = cat_model_1000.predict_proba(X_test)[:, 1]

# CALCULATE ROC-AUC
auc_1000 = roc_auc_score(
    Y_test,
    predictions_1000
)

print(f"CatBoost 1000 iterations ROC-AUC: {auc_1000:.5f}")


#COMPARE THE MODELS
print(f"\n MODEL COMPARISON")
print("-----------------------------")
print(f"Logistic Regression: {baseline_auc:.5f}")
print(f"CatBoost 500:        {cat_auc:.5f}")
print(f"CatBoost 1000:       {auc_1000:.5f}")


print("\n")


# TEST CATBOOST DEPTH FROM 6 TO 8
print(f"TEST CATBOOST DEPTH FROM 6 TO 8 EXPERIMENT")
print("-------------------------------------------------")

cat_model_depth8 = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=200
)

# BUILD THE MODEL
cat_model_depth8.fit(
    X_train,
    Y_train,
    cat_features=categorical_features,
    eval_set=(X_test, Y_test),
    early_stopping_rounds=100
)

# MAKE PROBABILITY PREDICTIONS
predictions_depth8 = cat_model_depth8.predict_proba(X_test)[:, 1]

# CALCULATE ROC-AUC
auc_depth8 = roc_auc_score(
    Y_test,
    predictions_depth8
)

print(f"\n CatBoost Depth 8 ROC-AUC: {auc_depth8:.5f}")


# COMPARE THE MODELS
print(f"\n MODEL COMPARISON")
print("-------------------------")
print(f"Logistic Regression: {baseline_auc:.5f}")
print(f"CatBoost 500:        {cat_auc:.5f}")
print(f"CatBoost 1000:       {auc_1000:.5f}")
print(f"CatBoost Depth 8:    {auc_depth8:.5f}")


print("\n")

# TEST CATBOOST DEPTH 6 + LEARNING RATE 0.03 + 2000 INTERATIONS
# Learning rate = 0.03
# Iterations = 2000
# Depth = 6
print(f"TEST CATBOOST DEPTH 6 + LEARNING RATE 0.03 + INTERATIONS = 2000")
print("------------------------------------------------------------------")

cat_model_lr03 = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=200
)

# BUILD THE MODEL
cat_model_lr03.fit(
    X_train,
    Y_train,
    cat_features=categorical_features,
    eval_set=(X_test, Y_test),
    early_stopping_rounds=100
)

# MAKE PROBABILITY PREDICTIONS
predictions_lr03 = cat_model_lr03.predict_proba(X_test)[:, 1]

# CALCULATE ROC-AUC
auc_lr03 = roc_auc_score(
    Y_test,
    predictions_lr03
)

print(f"\n CatBoost LR 0.03 ROC-AUC: {auc_lr03:.5f}")


# COMPARE THE MODELS
print(f"\n MODEL COMPARISON")
print("------------------------")
print(f"Logistic Regression:              {baseline_auc:.5f}")
print(f"CatBoost 500:                     {cat_auc:.5f}")
print(f"CatBoost 1000:                    {auc_1000:.5f}")
print(f"CatBoost Depth 8:                 {auc_depth8:.5f}")
print(f"CatBoost Depth 6,LR 0.03:         {auc_lr03:.5f}")



# FEATURE ENGINEERING EXPERIMENTS
print(f" FEATURE ENGINEERING EXPERIMENTS")
print("-----------------------------------")

# Income per Car
train["Income_Per_Car"] = (
    train["Annual_Income_USD"] /
    (train["Number_of_Cars_Owned"] + 1)
)

test["Income_Per_Car"] = (
    test["Annual_Income_USD"] /
    (test["Number_of_Cars_Owned"] + 1)
)


# Total Charging Availability
train["Total_Charging_Access"] = (
    train["Charging_Stations_Near_Home"] +
    train["Charging_Stations_Near_Work"]
)

test["Total_Charging_Access"] = (
    test["Charging_Stations_Near_Home"] +
    test["Charging_Stations_Near_Work"]
)


# Commute Relative to Charging Access
train["Commute_per_Charging"] = (
    train["Daily_Commute_km"] /
    (train["Total_Charging_Access"] + 1)
)

test["Commute_per_Charging"] = (
    test["Daily_Commute_km"] /
    (test["Total_Charging_Access"] + 1)
)


# Income + Environmental Concern
train["Income_Environmental_Score"] = (
    train["Annual_Income_USD"] *
    train["Environmental_Concern_Level"]
)

test["Income_Environmental_Score"] = (
    test["Annual_Income_USD"] *
    test["Environmental_Concern_Level"]
)


# REBUILD X AND Y
X_fe = train.drop(columns=["Will_Buy_EV", "id"])

Y_fe = train["Will_Buy_EV"].map({
    "No": 0,
    "Yes": 1
})


# RECREATE TRAIN, TEST / SPLIT
X_train_fe, X_test_fe, Y_train_fe, Y_test_fe = train_test_split(
    X_fe,
    Y_fe,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\n")

# BUILD NEW CATBOOST MODEL USING THE WINNING CATBOOST EXPERIMENT
print(f" FEATURE ENGINEERING INTERATION = 2000, LR = 0.03 EXPERIMENT")
print("----------------------------------------------------------------")

cat_model_fe = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=200
)


# TRAIN THE NEW CATBOOST MODEL
cat_model_fe.fit(
    X_train_fe,
    Y_train_fe,
    cat_features=categorical_features,
    eval_set=(X_test_fe, Y_test_fe),
    early_stopping_rounds=100
)


# MAKE PROBABILITY PREDICTIONS
predictions_fe = cat_model_fe.predict_proba(X_test_fe)[:, 1]


# CALCULATE ROC-AUC
auc_fe = roc_auc_score(
    Y_test_fe,
    predictions_fe
)

print(f"\n CatBoost with Feature Engineering ROC-AUC: {auc_fe:.5f}")


# COMPARE THE MODELS
print(f"\n MODEL COMPARISON")
print("-------------------------")
print(f"CatBoost 500:              {cat_auc:.5f}")
print(f"CatBoost 1000:             {auc_1000:.5f}")
print(f"CatBoost Depth 8:          {auc_depth8:.5f}")
print(f"CatBoost LR 0.03:          {auc_lr03:.5f}")
print(f"CatBoost + Feature Eng.:   {auc_fe:.5f}")


print("\n")


## CATBOOST EXPERIMENTS - L2 REGULARIZATION
print(f"CATBOOST L2 REGULARIZATION EXPERIMENT")
print("------------------------------------------")

cat_model_l2 = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=5,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=200
)

#BUILD THE MODEL
cat_model_l2.fit(
    X_train,
    Y_train,
    cat_features=categorical_features,
    eval_set=(X_test, Y_test),
    early_stopping_rounds=100
)

# MAKE PROBABILITY PREDICTIONS
predictions_l2 = cat_model_l2.predict_proba(X_test)[:, 1]


# CALCULATE ROC-AUC
auc_l2 = roc_auc_score(
    Y_test,
    predictions_l2
)

print(f"CatBoost L2=5 ROC-AUC: {auc_l2:.5f}")


# COMPARE THE MODELS
print(f"\n MODEL COMPARISON")
print("------------------------")
print(f"CatBoost 500:        {cat_auc:.5f}")
print(f"CatBoost 1000:       {auc_1000:.5f}")
print(f"CatBoost Depth 8:    {auc_depth8:.5f}")
print(f"CatBoost LR 0.03:    {auc_lr03:.5f}")
print(f"Feature Engineering: {auc_fe:.5f}")
print(f"CatBoost L2=5:       {auc_l2:.5f}")


print("\n")

# LIGHTGBM MODEL
# DEFINE LIGHTGBM FEATURES
X = train.drop(columns=["Will_Buy_EV", "id"])

Y = train["Will_Buy_EV"].map({
    "No": 0,
    "Yes": 1
})


# CONVERT CATEGORICAL FEATURES
for col in categorical_features:
    X[col] = X[col].astype("category")


# SPLIT TRAIN \ TEST
# Split Training and Validation Data
X_train_lgb, X_test_lgb, Y_train_lgb, Y_test_lgb = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# BUILD THE LIGHTGBM MODEL
lgb_model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="binary",
    metric="auc"
)


# TRAIN THE MODEL
lgb_model.fit(
    X_train_lgb,
    Y_train_lgb,
    categorical_feature=categorical_features,
    eval_set=[(X_test_lgb, Y_test_lgb)],
    callbacks=[
        lgb.early_stopping(
            stopping_rounds=100
        ),
        lgb.log_evaluation(200)
    ]
)



# MAKE PROBABILITY PREDICTIONS
lgb_predictions = lgb_model.predict_proba(
    X_test_lgb
)[:, 1]


# CALCULATE LIGHTGBM WITH ROC-AUC
lgb_auc = roc_auc_score(
    Y_test_lgb,
    lgb_predictions
)

print(f"\n LightGBM ROC-AUC: {lgb_auc:.5f}")



# COMPARE LIGHTGBM WITH PREVIOUS MODELS
print(f"\n MODEL COMPARISON")
print("-----------------------------")
print(f"Logistic Regression: {baseline_auc:.5f}")
print(f"CatBoost 500:        {cat_auc:.5f}")
print(f"CatBoost 1000:       {auc_1000:.5f}")
print(f"CatBoost Depth 8:    {auc_depth8:.5f}")
print(f"Feature Engineering: {auc_fe:.5f}")
print(f"CatBoost LR 0.03:    {auc_lr03:.5f}")
print(f"CatBoost L2=5:       {auc_l2:.5f}")
print(f"LightGBM:            {lgb_auc:.5f}")


print("\n")

#BLEND CATBOOST + LIGHTGBM
print(f"BLEND CATBOOST + LIGHTGBM")
print("-------------------------------")


# 70/30 BLEND
print(f"\n 70/30 BLEND")
print("--------------")

blend_70_30 = (
    0.7 * cat_predictions +
    0.3 * lgb_predictions
)

blend_auc_70_30 = roc_auc_score(
    Y_test,
    blend_70_30
)

print(f"CatBoost + LightGBM (70/30) ROC-AUC: {blend_auc_70_30:.5f}")


# 50/50 BLEND
print(f"\n 50/50 BLEND")
print("---------------")

blend_50_50 = (
    0.5 * cat_predictions +
    0.5 * lgb_predictions
)

blend_auc_50_50 = roc_auc_score(
    Y_test,
    blend_50_50
)

print(f"\n CatBoost + LightGBM (50/50) ROC-AUC: {blend_auc_50_50:.5f}")



# 80/20 BLEND
print(f"\n 80/20 BLEND")
print("--------------------")

blend_80_20 = (
    0.8 * cat_predictions +
    0.2 * lgb_predictions
)

blend_auc_80_20 = roc_auc_score(
    Y_test,
    blend_80_20
)

print(f"CatBoost + LightGBM (80/20) ROC-AUC: {blend_auc_80_20:.5f}")


# COMPARE THE RESULTS
print(f"\n BLENDING RESULTS")
print("-------------------------")
print(f"CatBoost:       {auc_lr03:.5f}")
print(f"LightGBM:       {lgb_auc:.5f}")
print(f"Blend 70/30:    {blend_auc_70_30:.5f}")
print(f"Blend 50/50:    {blend_auc_50_50:.5f}")
print(f"Blend 80/20:    {blend_auc_80_20:.5f}")

print("\n")


# 5-FOLD CROSS-VALIDATION
print("\n5-FOLD CROSS-VALIDATION")
print("-------------------------------")


# PREPARE THE FEATURES AND TARGET
x_cv = train.drop(columns=["Will_Buy_EV", "id"])

y_cv = train["Will_Buy_EV"].map({
    "No": 0,
    "Yes": 1
})


# CREATE STRATIFIED K-FOLD
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# CREATE LIST FOR FOLD SCORES
fold_scores = []


# TRAIN AND VALIDATE EACH FOLD
for fold, (train_idx, valid_idx) in enumerate(
    skf.split(x_cv, y_cv), start=1
):

    print(f"\nFold {fold}")


    # CREATE TRAINING DATA
    x_train_cv = x_cv.iloc[train_idx]
    x_valid_cv = x_cv.iloc[valid_idx]

    # CREATE VALIDATION TARGET
    y_train_cv = y_cv.iloc[train_idx]
    y_valid_cv = y_cv.iloc[valid_idx]


    # CREATE CATBOOST MODEL
    model_cv = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False
    )


    # TRAIN CATBOOST MODEL
    model_cv.fit(
        x_train_cv,
        y_train_cv,
        cat_features=categorical_features,
        eval_set=(x_valid_cv, y_valid_cv),
        early_stopping_rounds=100,
        verbose=False
    )


    # MAKE PROBABILITY PREDICTIONS
    predictions_cv = model_cv.predict_proba(
        x_valid_cv
    )[:, 1]


    # CALCULATE FOLD ROC-AUC
    fold_auc = roc_auc_score(
        y_valid_cv,
        predictions_cv
    )


    # STORE FOLD SCORE
    fold_scores.append(fold_auc)


    # PRINT FOLD RESULT
    print(f"Fold {fold}: ROC-AUC = {fold_auc:.5f}")


# DISPLAY CROSS-VALIDATION RESULTS
print("\nCROSS-VALIDATION RESULTS")
print("-----------------------------")

for i, score in enumerate(fold_scores, start=1):
    print(f"Fold {i}: {score:.5f}")


# CALCULATE MEAN ROC-AUC
mean_auc = np.mean(fold_scores)

# CALCULATE STANDARD DEVIATION
std_auc = np.std(fold_scores)


print(f"\nMean ROC-AUC: {mean_auc:.5f}")
print(f"Std ROC-AUC:  {std_auc:.5f}")


print("\n")

#XGBOOST EXPERIMENT
print(f" XGBOOST EXPERIMENT")
print("-----------------------------")

# SEPARATE FEATURES AND TARGET
X = train.drop(columns=["Will_Buy_EV", "id"])

Y = train["Will_Buy_EV"].map({
    "No": 0,
    "Yes": 1
})


# ENCODE CATEGORICAL FEATURES
preprocessor_xgb = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)



# TRANSFORM FEATURES
X_encoded = preprocessor_xgb.fit_transform(X)


# TRAIN / TEST SPLIT

X_train_xgb, X_test_xgb, Y_train_xgb, Y_test_xgb = train_test_split(
    X_encoded,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# BUILD XGBOOST MODEL
xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)



# TRAIN XGBOOST MODEL
xgb_model.fit(
    X_train_xgb,
    Y_train_xgb,
    eval_set=[
        (X_test_xgb, Y_test_xgb)
    ],
    verbose=200
)



# PREDICT PROBABILITIES
xgb_predictions = xgb_model.predict_proba(X_test_xgb)[:, 1]


# CALCULATE ROC-AUC

xgb_auc = roc_auc_score(
    Y_test_xgb,
    xgb_predictions
)

print(f"\n XGBoost ROC-AUC: {xgb_auc:.5f}")


# COMPARE THE MODELS
print(f"\n MODEL COMPARISON")
print("-----------------------------")

print(f"Logistic Regression: {baseline_auc:.5f}")
print(f"CatBoost 500:        {cat_auc:.5f}")
print(f"CatBoost 1000:       {auc_1000:.5f}")
print(f"CatBoost Depth 8:    {auc_depth8:.5f}")
print(f"Feature Engineering: {auc_fe:.5f}")
print(f"CatBoost LR 0.03:    {auc_lr03:.5f}")
print(f"CatBoost L2=5:       {auc_l2:.5f}")
print(f"LightGBM:            {lgb_auc:.5f}")
print(f"XGBoost:             {xgb_auc:.5f}")


print("\n")

# 5 - FOLD CROSSM - VALIDATION FOR XGBOOST
print(f"\n 5-Fold Cross-Validation for XGBoost")
print("--------------------------------------------")

# PREPARE FEATURES AND TARGET
X_cv = train.drop(columns=["Will_Buy_EV", "id"])

Y_cv = train["Will_Buy_EV"].map({
    "No": 0,
    "Yes": 1
})


# CREATE 5-FOLD STRATIFIED CROSS-VALIDATION
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_scores_xgb = []



# RUN 5-FOLD XGBOOST
for fold, (train_idx, valid_idx) in enumerate(
    skf.split(X_cv, Y_cv), start=1
):

    print(f"\nFold {fold}")


    #TRAIN / TEST SPLIT
    X_train_fold = X_cv.iloc[train_idx]
    X_valid_fold = X_cv.iloc[valid_idx]

    Y_train_fold = Y_cv.iloc[train_idx]
    Y_valid_fold = Y_cv.iloc[valid_idx]

    # ENCODE CATEGORICAL FEATURES
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            )
        ],
        remainder="passthrough"
    )

    X_train_encoded = preprocessor.fit_transform(
        X_train_fold
    )

    X_valid_encoded = preprocessor.transform(
        X_valid_fold
    )

    # BUILD XGBOOST MODEL
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )

    # TRAIN THE MODEL
    model.fit(
        X_train_encoded,
        Y_train_fold,
        eval_set=[
            (X_valid_encoded, Y_valid_fold)
        ],
        verbose=False
    )

    # MAKE PROBABILITY PREDICTIONS
    valid_predictions = model.predict_proba( X_valid_encoded)[:, 1]

    # CALCULATE ROC-AUC
    fold_auc = roc_auc_score(
        Y_valid_fold,
        valid_predictions
    )

    fold_scores_xgb.append(fold_auc)

    print(f"Fold {fold}: ROC-AUC = {fold_auc:.5f}")



# CALCULATE CROSS-VALIDATION RESULTS
print("\nXGBOOST 5-FOLD CROSS-VALIDATION")
print("------------------------------------")

for i, score in enumerate(
    fold_scores_xgb,
    start=1
):
    print(f"Fold {i}: {score:.5f}")


mean_auc_xgb = np.mean(fold_scores_xgb)
std_auc_xgb = np.std(fold_scores_xgb)

print(f"\nMean ROC-AUC: {mean_auc_xgb:.5f}")
print(f"Std ROC-AUC:  {std_auc_xgb:.5f}")


print("\n")



# FINAL XGBOOST 5-FOLD VALIDATION
print(f"FINAL XGBOOST 5-FOLD VALIDATION")
print("-----------------------------------")


# REMOVE UNSUCCESSFUL ENGINEERED FEATURES
engineered_features = [
    "Income_Per_Car",
    "Total_Charging_Access",
    "Commute_per_Charging",
    "Income_Environmental_Score"
]

train = train.drop(
    columns=engineered_features,
    errors="ignore"
)

test = test.drop(
    columns=engineered_features,
    errors="ignore"
)

# PREPARE FEATURES AND TARGET
X = train.drop(
    columns=["Will_Buy_EV", "id"])

Y = train["Will_Buy_EV"].map({
    "No": 0,
    "Yes": 1})

X_test = test.drop(
    columns=["id"])


# VERIFY FINAL FEATURE COUNT
print(f"\n Final training feature count:", X.shape[1])
print("Final training features:")
print(X.columns.tolist())


# CREATE 5-FOLD STRATIFIED CROSS-VALIDATION
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# CREATE STORAGE FOR OOF PREDICTIONS
oof_predictions = np.zeros(
    len(X)
)

fold_scores = []

fold_models = []
fold_preprocessors = []




# TRAIN XGBOOST USING 5-FOLD CROSS-VALIDATION
for fold, (train_idx, valid_idx) in enumerate(
    skf.split( X , Y ),
    start=1
):

    print(f"\n========== FOLD {fold} ==========")

    # SPLIT CURRENT FOLD
    X_train_fold = X.iloc[train_idx]
    X_valid_fold = X.iloc[valid_idx]

    Y_train_fold = Y.iloc[train_idx]
    Y_valid_fold = Y.iloc[valid_idx]

    # CREATE PREPROCESSOR
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            )
        ],
        remainder="passthrough"
    )


    # FIT PREPROCESSOR ON TRAINING FOLD ONLY
    X_train_encoded = preprocessor.fit_transform(
        X_train_fold
    )

    X_valid_encoded = preprocessor.transform(
        X_valid_fold
    )


    # BUILD XGBOOST MODEL
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )


    # TRAIN MODEL
    model.fit(
        X_train_encoded,
        Y_train_fold,
        eval_set=[
            (X_valid_encoded, Y_valid_fold)
        ],
        verbose=False
    )


    # VALIDATION PROBABILITIES
    valid_predictions = model.predict_proba(X_valid_encoded)[:, 1]


    # SAVE OOF PREDICTIONS
    oof_predictions[valid_idx] = valid_predictions


    # CALCULATE FOLD ROC-AUC
    fold_auc = roc_auc_score(
        Y_valid_fold,
        valid_predictions
    )

    fold_scores.append(
        fold_auc
    )

    print( f"Fold {fold}: ROC-AUC = {fold_auc:.5f}")


    # SAVE MODEL AND PREPROCESSOR
    fold_models.append(model)
    fold_preprocessors.append(preprocessor)


 # CALCULATE OVERALL OOF ROC-AUC
oof_auc = roc_auc_score(
    Y,
    oof_predictions
)

print(f"\nOVERALL OOF ROC-AUC")
print("-----------------------------")
print(f"OOF ROC-AUC: {oof_auc:.5f}")


# DISPLAY FOLD PERFORMANCE
print("\n5-FOLD XGBOOST RESULTS")
print("-----------------------------")

for i, score in enumerate(
    fold_scores,
    start=1
):
    print(f"Fold {i}: {score:.5f}")

print(
    f"\nMean ROC-AUC: "
    f"{np.mean(fold_scores):.5f}"
)

print(
    f"Std ROC-AUC: "
    f"{np.std(fold_scores):.5f}"
)


# MODEL COMPARISON — SINGLE VALIDATION EXPERIMENTS
print(f"\n CREATE MODEL COMPARISON DATA")
print("---------------------------------")

model_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "CatBoost 500",
        "CatBoost 1000",
        "CatBoost Depth 8",
        "CatBoost LR 0.03",
        "CatBoost L2=5",
        "LightGBM",
        "XGBoost"
    ],
    "ROC_AUC": [
        0.93796,
        0.94110,
        0.94149,
        0.94138,
        0.94158,
        0.94152,
        0.94152,
        0.94162
    ]
})

print(model_results)


# VISUALIZE MODEL COMPARISON
model_results_sorted = model_results.sort_values(
    "ROC_AUC"
)

plt.figure(figsize=(10, 7))

plt.barh(
    model_results_sorted["Model"],
    model_results_sorted["ROC_AUC"]
)

plt.title("Model ROC-AUC Comparison")
plt.xlabel("ROC-AUC")
plt.ylabel("Model")
plt.grid(axis="x",alpha=0.2)
plt.show()



# VISUALIZE 5-FOLD PERFORMANCE
fold_numbers = range(
    1,
    len(fold_scores) + 1
)

mean_auc = np.mean(fold_scores)

plt.figure(figsize=(9, 6))

plt.plot(
    fold_numbers,
    fold_scores,
    marker="o",
    linewidth=2,
    label="Fold ROC-AUC"
)

plt.axhline(
    mean_auc,
    linestyle="--",
    linewidth=2,
    label=f"Mean AUC = {mean_auc:.5f}"
)

plt.title("XGBoost 5-Fold Cross-Validation Performance")
plt.xlabel("Fold")
plt.ylabel("ROC-AUC")
plt.xticks(list(fold_numbers))
plt.legend()
plt.grid(alpha=0.2)
plt.show()


# CREATE ROC CURVE
fpr, tpr, thresholds = roc_curve(
    Y,
    oof_predictions
)

plt.figure(figsize=(9, 7))

plt.plot(
    fpr,
    tpr,
    linewidth=3,
    label=f"XGBoost AUC = {oof_auc:.5f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=2,
    label="Random Guess"
)

plt.title("XGBoost ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.2)
plt.show()

print(f"\n")

# FEATURE IMPORTANCE
print(f"\n # FEATURE IMPORTANCE ")
print("---------------------------")

# GET FEATURE NAMES
feature_names = (
    fold_preprocessors[0]
    .get_feature_names_out()
)



# CALCULATE AVERAGE FEATURE IMPORTANCE
print(f"\n # CALCULATE AVERAGE FEATURE IMPORTANCE")
print("--------------------------------------------------")

feature_importance_list = []

for model in fold_models:

    feature_importance_list.append(
        model.feature_importances_
    )

average_importance = np.mean(
    feature_importance_list,
    axis=0
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": average_importance
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=True
)



# VISUALIZE FEATURE IMPORTANCE
plt.figure(figsize=(10, 8))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.title("XGBoost Average Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.grid(axis="x",alpha=0.2)
plt.show()

print("\n")

# GENERATE NEW UNSEEN CUSTOMERS
print(f"\n 13. GENERATE NEW UNSEEN CUSTOMERS")
print("----------------------------------------")


# GENERATE NEW UNSEEN CUSTOMER DATA
np.random.seed(2026)

n_new = 1000

new_customers = pd.DataFrame({

    "Age": np.random.randint(
        train["Age"].min(),
        train["Age"].max() + 1,
        n_new
    ),

    "Annual_Income_USD": np.random.uniform(
        train["Annual_Income_USD"].min(),
        train["Annual_Income_USD"].max(),
        n_new
    ),

    "Daily_Commute_km": np.random.uniform(
        train["Daily_Commute_km"].min(),
        train["Daily_Commute_km"].max(),
        n_new
    ),

    "Number_of_Cars_Owned": np.random.randint(
        train["Number_of_Cars_Owned"].min(),
        train["Number_of_Cars_Owned"].max() + 1,
        n_new
    ),

    "Charging_Stations_Near_Home": np.random.randint(
        train["Charging_Stations_Near_Home"].min(),
        train["Charging_Stations_Near_Home"].max() + 1,
        n_new
    ),

    "Charging_Stations_Near_Work": np.random.randint(
        train["Charging_Stations_Near_Work"].min(),
        train["Charging_Stations_Near_Work"].max() + 1,
        n_new
    ),

    "Environmental_Concern_Level": np.random.randint(
        int(train["Environmental_Concern_Level"].min()),
        int(train["Environmental_Concern_Level"].max()) + 1,
        n_new
    ),

    "Gender": np.random.choice(
        train["Gender"].dropna().unique(),
        n_new
    ),

    "City_Type": np.random.choice(
        train["City_Type"].dropna().unique(),
        n_new
    ),

    "Current_Car_Type": np.random.choice(
        train["Current_Car_Type"].dropna().unique(),
        n_new
    ),

    "Home_Charging_Possible": np.random.choice(
        train["Home_Charging_Possible"].dropna().unique(),
        n_new
    ),

    "Subsidy_Available": np.random.choice(
        train["Subsidy_Available"].dropna().unique(),
        n_new
    ),

    "Range_Anxiety_Level": np.random.choice(
        train["Range_Anxiety_Level"].dropna().unique(),
        n_new
    )
})
print(f"\n NEW CUSTOMERS")
print("----------------------")

print(new_customers.head())
print("Shape:", new_customers.shape)


# CHECK UNSEEN CUSTOMER FEATURES
print("Training features:", X.columns.tolist())
print("Unseen features:", new_customers.columns.tolist())

assert list(X.columns) == list(new_customers.columns)


# PREDICT UNSEEN CUSTOMERS
print(f"\n PREDICT UNSEEN CUSTOMER PROBABILITIES")
print("--------------------------------------------")

# PREDICT UNSEEN CUSTOMER PROBABILITIES
unseen_predictions = []

for model, preprocessor in zip(
    fold_models,
    fold_preprocessors
):

    unseen_encoded = preprocessor.transform(
        new_customers
    )

    fold_prediction = model.predict_proba(
        unseen_encoded
    )[:, 1]

    unseen_predictions.append(
        fold_prediction
    )


 # AVERAGE PREDICTIONS FROM ALL FIVE MODELS
unseen_predictions = np.mean(
    unseen_predictions,
    axis=0
)

# ADD PREDICTIONS TO NEW CUSTOMER DATA
new_customers[ "EV_Purchase_Probability"] = unseen_predictions


# CREATE PREDICTION CATEGORIES
print(f"\n 15. CREATE PREDICTION CATEGORIES")
print("------------------------------------------")

new_customers["Prediction_Category"] = pd.cut(
    new_customers["EV_Purchase_Probability"],
    bins=[0, 0.25, 0.50, 0.75, 1],
    labels=[
        "Low Probability",
        "Moderate Probability",
        "High Probability",
        "Very High Probability"
    ],
    include_lowest=True
)



# VISUALIZE PREDICTED PROBABILITY DISTRIBUTION
plt.figure(figsize=(10, 6))

plt.hist(
    new_customers["EV_Purchase_Probability"],
    bins=30,
    density=True,
    alpha=0.7
)

mean_probability = (
    new_customers["EV_Purchase_Probability"].mean()
)

plt.axvline(
    mean_probability,
    linestyle="--",
    linewidth=2,
    label=f"Mean = {mean_probability:.3f}"
)

plt.title("Predicted EV Purchase Probability — Unseen Customers")
plt.xlabel("Probability of Buying an EV")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.2)
plt.show()



# ENVIRONMENTAL CONCERN VS PREDICTION
plt.figure(figsize=(10, 6))

sns.regplot(
    data=new_customers,
    x="Environmental_Concern_Level",
    y="EV_Purchase_Probability",
    scatter_kws={
        "alpha": 0.15
    },
    line_kws={
        "linewidth": 3
    },
    order=2
)

plt.title("Environmental Concern vs Predicted EV Purchase Probability")
plt.xlabel("Environmental Concern Level")
plt.ylabel("Predicted Probability")
plt.grid(alpha=0.2)
plt.show()



# INCOME VS PREDICTION
plt.figure(figsize=(10, 6))

sns.regplot(
    data=new_customers,
    x="Annual_Income_USD",
    y="EV_Purchase_Probability",
    scatter_kws={
        "alpha": 0.15
    },
    line_kws={
        "linewidth": 3
    },
    order=2
)

plt.title("Annual Income vs Predicted EV Purchase Probability")
plt.xlabel("Annual Income (USD)")
plt.ylabel("Predicted Probability")
plt.grid(alpha=0.2)
plt.show()



# SUBSIDY VS PREDICTION
plt.figure(figsize=(8, 6))

sns.boxplot(
    data=new_customers,
    x="Subsidy_Available",
    y="EV_Purchase_Probability"
)

plt.title("Subsidy Availability vs Predicted EV Purchase Probability")
plt.xlabel("Subsidy Available")
plt.ylabel("Predicted Probability")
plt.grid(axis="y",alpha=0.2)
plt.show()



# RANGE ANXIETY VS PREDICTION
plt.figure(figsize=(9, 6))

sns.boxplot(
    data=new_customers,
    x="Range_Anxiety_Level",
    y="EV_Purchase_Probability",
    order=["Low", "Medium", "High"]
)

plt.title("Range Anxiety vs Predicted EV Purchase Probability")
plt.xlabel("Range Anxiety Level")
plt.ylabel("Predicted Probability")
plt.grid(axis="y",alpha=0.2)
plt.show()

# DISPLAY TOP 20 PREDICTED CUSTOMERS
print(f"\n DISPLAY TOP 20 PREDICTED CUSTOMERS")
print("-----------------------------------------")

top_customers = new_customers.sort_values(
    "EV_Purchase_Probability",
    ascending=False
).head(20)

print(
    top_customers[
        [
            "Age",
            "Annual_Income_USD",
            "Environmental_Concern_Level",
            "Subsidy_Available",
            "Range_Anxiety_Level",
            "Home_Charging_Possible",
            "EV_Purchase_Probability"
        ]
    ]
)



# SAVE UNSEEN CUSTOMER PREDICTIONS
new_customers.to_csv(
    "unseen_customer_predictions.csv",
    index=False
)

print("Unseen customer predictions saved successfully!")

# CREATE STORAGE FOR FINAL TEST PREDICTIONS
final_test_predictions = np.zeros(
    len(X_test)
)


# GENERATE 5-FOLD TEST PREDICTIONS
for model, preprocessor in zip(
    fold_models,
    fold_preprocessors
):

    X_test_encoded = preprocessor.transform(
        X_test
    )

    fold_test_prediction = model.predict_proba(
        X_test_encoded
    )[:, 1]

    final_test_predictions += (
        fold_test_prediction /
        len(fold_models)
    )


# CHECK FINAL TEST PREDICTIONS
print(f"\n CHECK FINAL TEST PREDICTIONS")
print("------------------------------------")

print(
    f"Minimum probability: "
    f"{final_test_predictions.min():.6f}"
)

print(
    f"Maximum probability: "
    f"{final_test_predictions.max():.6f}"
)

print(
    f"Mean probability: "
    f"{final_test_predictions.mean():.6f}"
)



# CHECK TEST PREDICTIONS
print(f"\n CHECK TEST PREDICTIONS")
print("-----------------------------")

assert len(final_test_predictions) == len(test)
assert np.all(
    (final_test_predictions >= 0) &
    (final_test_predictions <= 1)
)

# CREATE KAGGLE SUBMISSION
print(f"\n CREATE KAGGLE SUBMISSION")
print("-------------------------------")

submission = pd.DataFrame({
    "id": test["id"],
    "Will_Buy_EV": final_test_predictions
})


# VALIDATE SUBMISSION
print(f"\n VALIDATE SUBMISSION")
print("----------------------------")

print("Submission shape:")
print(submission.shape)

print("\nSubmission columns:")
print(submission.columns.tolist())

print("\nMissing values:")
print(submission.isnull().sum())

print("\nFirst 5 rows:")
print(submission.head())


# SAVE FINAL KAGGLE SUBMISSION
submission.to_csv(
    "submission.csv",
    index=False
)

print("submission.csv created successfully!")



# FINAL SUBMISSION CHECK
print(f"\n FINAL SUBMISSION CHECK")
print("----------------------------")

final_check = pd.read_csv(
    "submission.csv"
)

print("Rows:", len(final_check))

print(
    "Columns:",
    list(final_check.columns)
)

print("\nMissing values:")
print(
    final_check.isnull().sum()
)

print("\nProbability range:")

print(
    "Minimum:",
    final_check["Will_Buy_EV"].min()
)

print(
    "Maximum:",
    final_check["Will_Buy_EV"].max()
)

print("\nFirst 5 rows:")
print(final_check.head())



# VISUALIZE FINAL KAGGLE TEST PREDICTIONS
plt.figure(figsize=(10, 6))

plt.hist(
    final_test_predictions,
    bins=30,
    density=True,
    alpha=0.7
)

mean_test_probability = (
    final_test_predictions.mean()
)

plt.axvline(
    mean_test_probability,
    linestyle="--",
    linewidth=2,
    label=f"Mean = {mean_test_probability:.3f}"
)

plt.title("Final XGBoost Predictions on Kaggle Test Data")
plt.xlabel("Predicted Probability of Buying an EV")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.2)
plt.show()