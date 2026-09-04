#IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
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
