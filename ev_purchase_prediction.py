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