import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.title("Agentic EDA Tool")
st.write("Upload a CSV file to begin automated exploratory data analysis.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    st.subheader("Basic Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Summary Statistics (Numeric Columns Only)")
    numeric_df = df.select_dtypes(include=['number'])
    st.write(numeric_df.describe())

    st.subheader("Correlation Matrix (Numeric Columns Only)")
    corr = numeric_df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Categorical Feature Analysis")
    categorical_df = df.select_dtypes(include=['object', 'category'])
    for col in categorical_df.columns:
        st.write(f"Value Counts for {col}:")
        st.write(df[col].value_counts())
        fig, ax = plt.subplots()
        sns.countplot(x=col, data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    if 'price' in df.columns:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['price'], kde=True, ax=ax)
        st.pyplot(fig)

        if 'size' in df.columns:
            st.subheader("Price vs Size")
            fig, ax = plt.subplots()
            sns.scatterplot(x='size', y='price', data=df, ax=ax)
            st.pyplot(fig)

        if 'age' in df.columns:
            st.subheader("Price vs Age")
            fig, ax = plt.subplots()
            sns.scatterplot(x='age', y='price', data=df, ax=ax)
            st.pyplot(fig)

        if 'bedrooms' in df.columns:
            st.subheader("Price vs Bedrooms")
            fig, ax = plt.subplots()
            sns.boxplot(x='bedrooms', y='price', data=df, ax=ax)
            st.pyplot(fig)

    st.success("EDA completed. Scroll through the sections above to view insights.")
