Air Pollution Trend Analysis and Pattern

Project Overview
Air pollution is a critical environmental and public health issue. This project focuses on analyzing long-term air quality trends across the United States using real-world government data. By applying both supervised and unsupervised machine learning techniques, the project aims to identify pollution patterns, predict air quality levels, and classify regions based on pollution severity.
The analysis is performed on data obtained from the National Environmental Health Tracking Network, making the project highly relevant for real-world environmental analytics and policy-driven insights.

Dataset
Source: National Environmental Health Tracking Network (via Data.gov)
Data Type: Real-world air quality measurements
Key Attributes:
  ReportYear
  StateName
  CountyName
  MeasureName
  MeasureType
  Air Quality Value (Target Variable)

Tools & Technologies
Programming Language: Python
Libraries Used:
  Pandas
  NumPy
  Matplotlib
  Scikit-learn

Data Preprocessing
- Removed records with missing target values to ensure model reliability
- Dropped irrelevant columns to reduce noise and dimensionality
- Converted data types for numerical consistency
- Encoded categorical variables using Label Encoding
- Applied feature scaling for clustering and PCA

Machine Learning Techniques Used
  Supervised Learning
    - Linear Regression
    - Multiple Linear Regression
    Evaluation Metrics:
      MAE (Mean Absolute Error)
      MSE (Mean Squared Error)
      RMSE (Root Mean Squared Error)
      R² Score
  Unsupervised Learning
    - K-Means Clustering
      Identified low, medium, and high pollution regions
    - Principal Component Analysis (PCA)
      Reduced dimensionality while retaining maximum variance

Visualizations
- Scatter plots for trend analysis
- Regression line plots (actual vs predicted values)
- Clustered scatter plots for pollution grouping
- PCA 2D plots for dimensionality reduction insights

Key Outcomes
- Identified long-term air pollution trends across U.S. regions
- Successfully predicted air quality values using regression models
- Grouped regions into pollution risk categories using clustering
- Simplified complex datasets using PCA for better interpretability

Future Scope
- Implement advanced ML models (Random Forest, SVR, Gradient Boosting)
- Apply time-series forecasting techniques (ARIMA, LSTM)
- Integrate geospatial visualization using GIS tools
- Extend the system to real-time air quality monitoring
