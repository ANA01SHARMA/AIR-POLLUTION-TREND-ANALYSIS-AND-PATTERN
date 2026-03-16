import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv(
    "D:/SEM_5 STUDY MATERIAL/INT234_PREDICTIVE ANALYTICS/CA2/"
    "Air_Quality_Measures_on_the_National_Environmental_Health_Tracking_Network.csv"
)

# ==============================
# BASIC INSPECTION
# ==============================

print("===== DATASET SHAPE =====")
print(df.shape, "\n")

print("===== COLUMN NAMES =====")
print(df.columns, "\n")

print("===== DATASET INFO =====")
print(df.info(), "\n")

print("===== STATISTICAL SUMMARY =====")
print(df.describe(), "\n")

print("===== MISSING VALUES =====")
print(df.isnull().sum(), "\n")

# ==============================
# DATA CLEANING & PREPROCESSING
# ==============================

df = df.dropna(subset=['Value'])

df = df.drop(columns=[
    'Unit',
    'UnitName',
    'DataOrigin',
    'StratificationLevel'
])

df['ReportYear'] = df['ReportYear'].astype(int)
df['Value'] = df['Value'].astype(float)

le = LabelEncoder()
df['StateName'] = le.fit_transform(df['StateName'])
df['CountyName'] = le.fit_transform(df['CountyName'])
df['MeasureName'] = le.fit_transform(df['MeasureName'])
df['MeasureType'] = le.fit_transform(df['MeasureType'])

print("===== CLEANED DATA (FIRST 5 ROWS) =====")
print(df.head(), "\n")

print("===== FINAL DATASET INFO =====")
print(df.info(), "\n")

print("===== FINAL DATASET SHAPE =====")
print(df.shape)

# ==============================
# SUPERVISED MODEL 1:
# LINEAR REGRESSION
# ==============================

X = df[['ReportYear']]
y = df['Value']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

print("\n===== LINEAR REGRESSION RESULTS =====")
print("MSE :", mean_squared_error(y_test, y_pred))
print("R²  :", r2_score(y_test, y_pred), "\n")

plt.figure(figsize=(8, 5))
plt.scatter(
    X_test, y_test,
    color="#1f77b4",      # blue
    alpha=0.6,
    label="Actual Values"
)
plt.plot(
    X_test, y_pred,
    color="#ff7f0e",      # orange
    linewidth=2,
    label="Regression Line"
)
plt.xlabel("Report Year")
plt.ylabel("Air Quality Value")
plt.title("Linear Regression: Year vs Air Quality")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()

# ==============================
# SUPERVISED MODEL 2:
# MULTIPLE LINEAR REGRESSION
# ==============================

X = df[['ReportYear', 'StateName', 'CountyName', 'MeasureName', 'MeasureType']]
y = df['Value']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)

y_pred = mlr_model.predict(X_test)

print("===== MULTIPLE LINEAR REGRESSION RESULTS =====")
print("MAE  :", mean_absolute_error(y_test, y_pred))
print("MSE  :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²   :", r2_score(y_test, y_pred), "\n")

plt.figure(figsize=(7, 6))
plt.scatter(
    y_test, y_pred,
    color="#2ca02c",      # green
    alpha=0.6
)
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.title("Multiple Linear Regression: Actual vs Predicted")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()

# ==============================
# UNSUPERVISED MODEL 1:
# K-MEANS CLUSTERING
# ==============================

X_cluster = df[['ReportYear', 'Value']]
X_scaled = StandardScaler().fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("===== K-MEANS CLUSTER COUNTS =====")
print(df['Cluster'].value_counts(), "\n")

plt.figure(figsize=(8, 6))
plt.scatter(
    df['ReportYear'],
    df['Value'],
    c=df['Cluster'],
    cmap="viridis",      # attractive colormap
    alpha=0.7
)
plt.xlabel("Year")
plt.ylabel("Air Quality Value")
plt.title("K-Means Clustering of Air Quality Data")
plt.colorbar(label="Cluster")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()

# ==============================
# UNSUPERVISED MODEL 2:
# PRINCIPAL COMPONENT ANALYSIS
# ==============================

X_pca = df.drop(columns=['Value', 'Cluster'])
X_scaled = StandardScaler().fit_transform(X_pca)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    principal_components,
    columns=['Principal Component 1', 'Principal Component 2']
)

print("===== PCA EXPLAINED VARIANCE RATIO =====")
print(pca.explained_variance_ratio_, "\n")

plt.figure(figsize=(8, 6))
plt.scatter(
    pca_df['Principal Component 1'],
    pca_df['Principal Component 2'],
    color="#9467bd",     # purple
    alpha=0.6
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA – Air Quality Dataset")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()
