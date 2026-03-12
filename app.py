import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.title("❤️ Heart Disease Prediction App")

st.write("Upload the heart disease dataset to train and test the model.")

# Upload dataset
uploaded_file = st.file_uploader("Upload Heart Disease Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Check if target column exists
    if "target" not in df.columns:
        st.error("Dataset must contain a 'target' column.")
    else:

        # Visualization
        st.subheader("Heart Disease Distribution")

        fig1, ax1 = plt.subplots()
        sns.countplot(x="target", data=df, ax=ax1)
        ax1.set_title("Heart Disease Distribution")
        st.pyplot(fig1)

        st.subheader("Correlation Heatmap")

        fig2, ax2 = plt.subplots(figsize=(10,6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax2)
        ax2.set_title("Feature Correlation")
        st.pyplot(fig2)

        # Data Processing
        X = df.drop("target", axis=1)
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model Training
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Model Testing
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        st.subheader("Model Accuracy")
        st.success(f"Accuracy: {accuracy:.2f}")

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))
