# ---------------------- ANALYSIS & MODEL TRAINING ----------------------

import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import joblib

warnings.filterwarnings('ignore')

# Load and merge data
def load_and_merge_data():
    customer = pd.read_csv(r"C:\Users\kkt\Downloads\customer_churn\data\Customer_Info.csv")
    location = pd.read_csv(r"C:\Users\kkt\Downloads\customer_churn\data\Location_Data.csv")
    service = pd.read_csv(r"C:\Users\kkt\Downloads\customer_churn\data\Online_Services.csv").replace({'Yes': 1, 'No': 0})
    payment = pd.read_csv(r"C:\Users\kkt\Downloads\customer_churn\data\Payment_Info.csv").rename(columns={'monthly_ charges': 'monthly_charges'})
    option = pd.read_csv(r"C:\Users\kkt\Downloads\customer_churn\data\Service_Options.csv").replace({'Yes': 1, 'No': 0})
    status = pd.read_csv(r"C:\Users\kkt\Downloads\customer_churn\data\Status_Analysis.csv")

    # Merge
    wide_customer = customer.merge(service, on="customer_id").merge(payment, on="customer_id")
    wide_customer = wide_customer.merge(option, on="customer_id").merge(status, on="customer_id")
    return wide_customer

# Feature engineering
def preprocess_data(df):
    x = df.copy()
    x['tech_service'] = x.online_security + x.online_backup + x.device_protection + x.premium_tech_support
    x['streaming_service'] = x.streaming_tv + x.streaming_movies + x.streaming_music
    x = x[['age', 'number_of_dependents', 'internet_type', 'phone_service_x', 'streaming_service', 
           'tech_service', 'contract', 'unlimited_data', 'number_of_referrals', 'satisfaction_score']]
    x = pd.get_dummies(x, columns=['contract', 'internet_type'], drop_first=True, dtype=int)
    return x

# Train model
def train_model():
    data = load_and_merge_data()
    x = preprocess_data(data)
    y = data[['churn_value']].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train.values.ravel())

    # Save model and feature names
    joblib.dump((model, x.columns.tolist(), x_test, y_test), 'churn_model.pkl')
    return model, x.columns.tolist(), x_test, y_test

# Run training
model, feature_names, x_test, y_test = train_model()

# ---------------------- STREAMLIT APP ----------------------

import streamlit as st
import os

st.set_page_config(page_title="Customer Churn App", layout="wide")
st.title("📊 Customer Churn & CLTV Prediction App")

# Sidebar Navigation
st.sidebar.header("🔍 Navigation")
app_mode = st.sidebar.radio("Choose App Mode:", ["Prediction", "EDA / Insights", "About"])

# Load data function
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\kkt\Downloads\customer_churn\data\Customer_Info.csv")

data = load_data()



# User input form
def user_input_features():
    st.sidebar.subheader("📋 Manual Input")
    columns_to_drop = [col for col in ['Churn', 'CLTV', 'CustomerID'] if col in data.columns]
    feature_cols = data.drop(columns=columns_to_drop).columns
    user_data = {}
    for col in feature_cols:
        if data[col].dtype == 'object':
            user_data[col] = st.sidebar.selectbox(f"{col}", sorted(data[col].dropna().unique()))
        else:
            if col in ['number_of_dependents', 'age']:
                # Enforce integer input for age and number_of_dependents
                user_data[col] = st.sidebar.number_input(
                    f"{col}",
                    min_value=int(data[col].min()),
                    max_value=int(data[col].max()),
                    value=int(data[col].mean()),
                    step=1
                )
            else:
                # Allow float for other numeric columns (if any)
                user_data[col] = st.sidebar.number_input(
                    f"{col}",
                    min_value=float(data[col].min()),
                    max_value=float(data[col].max()),
                    value=float(data[col].mean())
                )
    return pd.DataFrame([user_data])



# ===================== APP MODE: PREDICTION =====================
if app_mode == "Prediction":
    input_mode = st.sidebar.radio("Select Input Mode:", ["Manual Entry", "Upload CSV"])

    if input_mode == "Manual Entry":
        input_df = user_input_features()
    else:
        uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type=["csv"])
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully.")
        else:
            st.warning("Please upload a CSV file.")
            st.stop()

    # Load model and feature names
    model_path = "churn_model.pkl"
    if os.path.exists(model_path):
        model, expected_features, x_test, y_test = joblib.load(model_path)
    else:
        st.error("Model file not found. Please run the training script first.")
        st.stop()

    # Prepare input data for prediction
    st.subheader("🔮 Prediction Results")
    if 'input_df' in locals():
        # Apply the same preprocessing as training
        input_df_processed = pd.get_dummies(input_df, dtype=int)

        # Align columns with training data
        missing_cols = set(expected_features) - set(input_df_processed.columns)
        for col in missing_cols:
            input_df_processed[col] = 0
        extra_cols = set(input_df_processed.columns) - set(expected_features)
        input_df_processed = input_df_processed.drop(columns=extra_cols, errors='ignore')

        # Ensure column order matches training data
        input_df_processed = input_df_processed[expected_features]

        # Predict
        predictions = model.predict(input_df_processed)
        prediction_proba = model.predict_proba(input_df_processed)
        input_df['Churn_Prediction'] = predictions
        input_df['Churn_Probability'] = prediction_proba[:, 1]
        st.dataframe(input_df.style.highlight_max(axis=0, color="lightgreen"))
        st.success("Prediction complete.")

    # Show model performance
    with st.expander("📈 Model Performance on Test Set"):
        y_pred = model.predict(x_test)
        st.markdown("**Confusion Matrix:**")
        st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))

        st.markdown("**Classification Report:**")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

# ===================== APP MODE: EDA / INSIGHTS =====================
elif app_mode == "EDA / Insights":
    st.header("🔍 Exploratory Data Analysis")

    # Load additional data for EDA
    customer = pd.read_csv(r"C:\Users\kkt\Downloads\customer_churn\data\Customer_Info.csv")
    service = pd.read_csv(r"C:\Users\kkt\Downloads\customer_churn\data\Online_Services.csv").replace({'Yes': 1, 'No': 0})
    status = pd.read_csv(r"C:\Users\kkt\Downloads\customer_churn\data\Status_Analysis.csv")

    # Age + gender dist
    with st.expander("Customer Age and Gender Distribution"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(data=customer, x='age', hue='gender', multiple='stack',
                     shrink=0.9, alpha=0.85, ax=axes[0], palette="viridis")
        age_group = customer[['under_30', 'senior_citizen']].replace({'Yes': 1, 'No': 0})
        age_group['30-65'] = 1 - (age_group.under_30 + age_group.senior_citizen)
        age_group = age_group.sum().reset_index(name='count')
        axes[1].pie(age_group['count'], labels=age_group['index'], autopct='%1.1f%%',
                    colors=["#ff9999", "#66b3ff", "#99ff99"], startangle=90,
                    wedgeprops={'edgecolor': 'black'})
        st.pyplot(fig)

    # Correlation heatmap of services
    with st.expander("Service Correlation Heatmap"):
        service_matrix = service.replace({'Yes': 1, 'No': 0})
        service_matrix = pd.get_dummies(service_matrix, columns=['internet_type'], dtype=int)
        corr_matrix = service_matrix.drop(['customer_id'], axis=1).corr()
        fig = plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, cmap="mako", annot=True, fmt=".2f", linewidths=0.5,
                    vmin=-1, vmax=1, cbar=True, square=True, annot_kws={"size": 8})
        st.pyplot(fig)

    # Churn category pie + boxplot
    with st.expander("Churn Categories & Satisfaction"):
        churn = status[status.customer_status == 'Churned'].drop(columns=['customer_status', 'churn_value'])
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        churn_cat = churn.groupby('churn_category').size().reset_index(name='count')
        ax[0].pie(churn_cat['count'], labels=churn_cat['churn_category'], autopct='%1.1f%%',
                  colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0'],
                  startangle=140, wedgeprops={'edgecolor': 'black'})
        sns.boxplot(data=churn, x='churn_category', y='satisfaction_score',
                    ax=ax[1], palette='coolwarm')
        st.pyplot(fig)

# ===================== APP MODE: ABOUT =====================
elif app_mode == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    This application was built to help predict telecom customer churn using logistic regression.  
    It also provides powerful visual insights into customer behavior through EDA.
    
    **Features**:
    - Manual or CSV input
    - Live predictions
    - EDA plots
    - Confusion matrix & classification report
    """)