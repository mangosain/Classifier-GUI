import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

import time

def main():
    st.title("Visual Workbench")
    st.markdown("This web app provides a GUI for basic data preprocessing and model training using Python.")

    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        @st.cache_data(persist=False)
        def load_data(file):
            data = pd.read_csv(file)
            label = LabelEncoder()
            for column in data.columns:
                if data[column].dtype == 'object' or data[column].dtype.name == 'category':
                    data[column] = label.fit_transform(data[column].astype(str))
            return data

        df = load_data(uploaded_file)

        st.sidebar.checkbox("Show raw data", value=False, key="show_raw_data")
        if st.session_state.get("show_raw_data", False):
            st.subheader("Raw Data")
            st.write(df)

        st.sidebar.checkbox("Show statistics", value=False, key="show_statistics")
        if st.session_state.get("show_statistics", False):
            st.subheader("Data Overview")
            st.write(df.describe())
            st.write("Number of rows:", df.shape[0])
            st.write("Number of columns:", df.shape[1])

        st.sidebar.checkbox("EDA", value=False, key="eda")
        if st.session_state.get("eda", False):
            st.subheader("Missing Values")
            st.write(df.isnull().sum())
            st.subheader("Data Types")
            st.write(df.dtypes)
            st.subheader("Correlation Matrix")
            st.write(df.corr())
            st.subheader("Heatmap of Correlation Matrix")
            plt.figure(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
            st.pyplot(plt)
            st.subheader("Histograms of Numerical Features")
            df.hist(bins=30, color='b', figsize=(22, 22))
            st.pyplot(plt)

        st.sidebar.divider()
        st.sidebar.subheader("Data Splitting")

        target_variable = st.sidebar.selectbox("Select Target Variable", df.columns)

        col1, col2 = st.sidebar.columns(2)
        with col1:
            test_size = st.number_input("Test Size (%)", min_value=10, max_value=90, value=20, key="test_size_input")
            shuffle = st.checkbox("Shuffle Data", value=False, key="shuffle_data")
        with col2:
            random_state = st.number_input("Random State", min_value=0, max_value=100, value=42, key="random_state_input")
            stratify = st.checkbox("Stratify Split", value=False, key="stratify_split")

        scale_data = st.sidebar.checkbox("Scale Features (StandardScaler)", value=False, key="scale_features")

        # Model selection
        st.sidebar.divider()
        st.sidebar.subheader("Select Model")
        model_options = {
            "SVM": SVC,
            "Random Forest": RandomForestClassifier,
            "Logistic Regression": LogisticRegression
        }
        selected_model = st.sidebar.selectbox("Model", list(model_options.keys()), key="model_select")

        model_params = {}
        if selected_model == "SVM":
            model_params["C"] = st.sidebar.number_input("C (Regularization)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
            model_params["kernel"] = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
            model_params["gamma"] = st.sidebar.selectbox("Gamma", ["scale", "auto"])
        elif selected_model == "Random Forest":
            model_params["n_estimators"] = st.sidebar.number_input("n_estimators", min_value=10, max_value=500, value=100, step=10)
            model_params["max_depth"] = st.sidebar.number_input("max_depth", min_value=1, max_value=50, value=10, step=1)
        elif selected_model == "Logistic Regression":
            model_params["C"] = st.sidebar.number_input("C (Inverse Regularization)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
            model_params["max_iter"] = st.sidebar.number_input("max_iter", min_value=50, max_value=1000, value=100, step=10)

        train_btn = st.sidebar.button("Train Model", use_container_width=True)

        @st.cache_data(persist=True)
        def split_data(df, target, test_size, random_state, shuffle, stratify_enabled, scale=False):
            y = df[target]
            X = df.drop(columns=[target])
            stratify_param = y if stratify_enabled else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100.0, random_state=random_state,
                shuffle=shuffle, stratify=stratify_param
            )
            if scale:
                scaler = StandardScaler()
                X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
                X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
            return X_train, X_test, y_train, y_test

        if train_btn:
            st.session_state["interrupt"] = False
            X_train, X_test, y_train, y_test = split_data(df, target_variable, test_size, random_state, shuffle, stratify, scale_data)

            st.session_state["X_train"] = X_train
            st.session_state["X_test"] = X_test
            st.session_state["y_train"] = y_train
            st.session_state["y_test"] = y_test

            progress_bar = st.progress(0, text="Training model...")

            if st.button("Interrupt Training"):
                st.session_state["interrupt"] = True

            for i in range(100):
                if st.session_state.get("interrupt", False):
                    st.warning("Training interrupted!")
                    progress_bar.empty()
                    st.stop()
                time.sleep(0.01)
                progress_bar.progress(i + 1, text="Training model...")

            ModelClass = model_options[selected_model]
            if selected_model == "SVM":
                model = ModelClass(C=model_params["C"], kernel=model_params["kernel"], gamma=model_params["gamma"], probability=True)
            elif selected_model == "Random Forest":
                model = ModelClass(n_estimators=int(model_params["n_estimators"]), max_depth=int(model_params["max_depth"]))
            elif selected_model == "Logistic Regression":
                model = ModelClass(C=model_params["C"], max_iter=int(model_params["max_iter"]))

            model.fit(X_train, y_train)

            st.session_state["model"] = model
            st.session_state["model_trained"] = True
            progress_bar.empty()
            st.success(f"{selected_model} trained successfully!")

        if st.session_state.get("model_trained", False):
            model = st.session_state["model"]
            X_test = st.session_state["X_test"]
            y_test = st.session_state["y_test"]
            y_pred = model.predict(X_test)

            st.header("Model Metrics")
            report = classification_report(y_test, y_pred, output_dict=True)
            acc = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')

            st_cols = st.columns(3)
            st_cols[0].metric("Accuracy", f"{acc:.2f}")
            st_cols[1].metric("Recall", f"{recall:.2f}")
            st_cols[2].metric("Precision", f"{precision:.2f}")
            
            
            st.subheader("Classification Report")
            st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='Blues'))

            st.subheader("Visualizations")
            if st.checkbox("Show Confusion Matrix"):
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title("Confusion Matrix")
                st.pyplot(plt)

            if st.checkbox("Show ROC Curve"):
                if hasattr(model, "predict_proba") and y_test.nunique() == 2:
                    y_score = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_score)
                    auc = roc_auc_score(y_test, y_score)
                    plt.figure(figsize=(6, 4))
                    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
                    plt.plot([0, 1], [0, 1], '--', color='gray')
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("ROC Curve")
                    plt.legend()
                    st.pyplot(plt)
                else:
                    st.warning("ROC curve only available for binary classification models with predict_proba.")

            st.subheader("Make a Prediction")
            input_cols = st.session_state["X_train"].columns
            user_input = {}
            with st.form("prediction_form"):
                for col in input_cols:
                    val = st.number_input(f"{col}", value=0.0, key=f"pred_{col}")
                    user_input[col] = val
                submitted = st.form_submit_button("Predict")
            if submitted:
                input_df = pd.DataFrame([user_input])
                pred = model.predict(input_df)[0]
                st.success(f"Predicted class: {pred}")

    else:
        st.info("Please upload a CSV file to get started.")

if __name__ == "__main__":
    main()
