 Customer Purchase Prediction AI
 Live Demo: https://huggingface.co/spaces/adhd12345/customer-purchase-prediction

[Python](https://img.shields.io/badge/Python-3.9-blue)
[Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)
[Gradio](https://img.shields.io/badge/Frontend-Gradio-orange)

 Project Overview
This project is a Machine Learning application designed to predict whether a customer will purchase a product based on their demographic and behavioral data. It helps businesses identify high-value customers and optimize marketing strategies.

Key Goal: Solve the business problem of "Targeting the Right Customer" to save ad budget and increase revenue.

 🌟 Key Features
- AI Prediction Engine: Uses a Random Forest Classifier (Accuracy: ~94%) to predict purchase probability.
- Interactive Dashboard: Visualizes customer data against market trends using Plotly interactive charts.
- Intelligent Business Advice: Automatically generates actionable marketing strategies (e.g., "Send Discount Coupon" vs. "Don't Spend Ad Budget") based on predictions.
- Visual Segmentation: Plots the customer on a "Spending Score vs. Income" graph to identify their market segment.

Tech Stack
  Language: Python
  Machine Learning: Scikit-Learn (Random Forest, Logistic Regression)
  Data Processing: Pandas, NumPy
  Visualization: Plotly, Matplotlib
  Web App Framework: Gradio
  Deployment:Hugging Face Spaces

 Project Structure
 `app.py`: The main application script containing the UI and logic.
 `customer_model.pkl`: The trained Machine Learning model.
 `scaler.pkl`: The data scaler ensuring accurate predictions.
 `Project_Notebook.ipynb`: Jupyter Notebook containing data analysis, model training, and evaluation metrics.
 `requirements.txt`: List of dependencies required to run the app.

Model Insights
The model was trained on the "Online Customer Dataset". After comparing multiple algorithms, **Random Forest** was selected for its superior Precision and Recall scores compared to Logistic Regression and KNN.

Top Predictors: Annual Income, Spending Score.
Target Audience: High Income + High Spending Score cluster.


Created by Ashish
