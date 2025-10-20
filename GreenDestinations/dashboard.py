import streamlit as st
from PIL import Image
import os
import pandas as pd
df = pd.read_csv('greendestination.csv')   # Make sure your file name matches exactly
print(" Data Loaded Successfully!\n")

logo_path = r"C:\Users\likit\OneDrive\Desktop\GreenDestinations\greendestination+logo.png"

if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, width=200)  # adjust width as needed
else:
    st.warning("Logo image not found at the specified path.")



    st.text("Logo missing")


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

st.title("📊 Employee Attrition Analysis - Green Destinations")
st.markdown("Analyze and visualize employee attrition patterns with key factors like **Age**, **Years at Company**, and **Monthly Income**.")

uploaded_file = st.file_uploader("📁 Upload your employee dataset (.csv)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.replace(' ', '_')

    st.success("✅ File uploaded successfully!")
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    if 'Attrition' in df.columns:
        attrition_rate = (df['Attrition'].value_counts(normalize=True).get('Yes', 0)) * 100
        st.metric("📉 Attrition Rate", f"{attrition_rate:.2f}%")

    st.subheader("📊 Visual Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Attrition Distribution**")
        fig, ax = plt.subplots()
        df['Attrition'].value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue', 'lightcoral'], startangle=90, ax=ax)
        plt.ylabel('')
        st.pyplot(fig)

    with col2:
        st.write("**Gender vs Attrition**")
        if 'Gender' in df.columns:
            fig, ax = plt.subplots()
            sns.countplot(x='Gender', hue='Attrition', data=df, ax=ax)
            st.pyplot(fig)

    st.write("---")
    st.subheader("📈 Relationship with Key Factors")

    key_cols = ['Age', 'YearsAtCompany', 'MonthlyIncome']
    available_features = [col for col in key_cols if col in df.columns]

    for feature in available_features:
        fig, ax = plt.subplots()
        sns.boxplot(x='Attrition', y=feature, data=df, ax=ax)
        plt.title(f"{feature} vs Attrition")
        st.pyplot(fig)

    st.write("---")
    st.subheader("🤖 Predictive Attrition Model")

    df['Attrition_flag'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    X = df[available_features]
    y = df['Attrition_flag']

    if len(available_features) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("**Model Performance**")
        st.text(classification_report(y_test, y_pred))

        coef_df = pd.DataFrame({
            'Feature': available_features,
            'Coefficient': model.coef_[0]
        }).sort_values(by='Coefficient', ascending=False)

        fig, ax = plt.subplots()
        sns.barplot(x='Coefficient', y='Feature', data=coef_df, ax=ax)
        plt.title("Feature Impact on Attrition (Positive → Higher Risk)")
        st.pyplot(fig)

        st.write("---")
        st.subheader("💡 Insights Summary")

        insights = []
        if 'Age' in available_features:
            insights.append("Younger employees show higher attrition rates.")
        if 'YearsAtCompany' in available_features:
            insights.append("Employees with fewer years at the company are more likely to leave.")
        if 'MonthlyIncome' in available_features:
            insights.append("Lower-income employees tend to leave more often.")

        for point in insights:
            st.markdown(f"- {point}")

    else:
        st.warning("⚠️ Not enough numeric features to build a model.")

else:
    st.info("👆 Upload a CSV file to begin the analysis.")
