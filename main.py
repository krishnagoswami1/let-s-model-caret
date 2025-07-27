import streamlit as st
import pandas as pd
from pycaret.classification import setup as setup_classification, compare_models as compare_models_classification, pull as pull_classification
from pycaret.regression import setup as setup_regression, compare_models as compare_models_regression, pull as pull_regression
st.set_page_config(title="🚀 Let's Model", layout="wide")
st.title("🤖Let's Model")

st.write("Upload your csv file: ")
uploaded_file = st.file_uploader("Upload your csv file: ", type = ['csv'])
if uploaded_file:
  df = pd.read_csv(uploaded_file)
  st.write(df.head())

  target = st.selectbox("Select your target column: ", df.columns)
  if target:
    task_type = 'classification' if df[target].nunique() <= 10 or df[target].dtype == 'object' else 'regression'
    st.write(f"🔍 Detected task type: {task_type}")

    if st.button("⚙️Generate Model"):
      with st.spinner("Running PyCaret.... This may take a moment 🔃"):
        if task_type == 'classification':
          setup_classification(data = df, target = target, session_id = 123, silent = True, verbose = False)
          best_model = compare_models_classification()
          results = pull_classification()
        else:
          setup_regression(data = df, target = target, session_id = 123, silent = True, verbose = False)
          best_model = compare_models_regression()
          results = pull_regression()
        
      st.success("✔️Model Training Complete!")
      st.subheader("📈 Model Leaderboard")
      st.dataframe(results)
      st.subheader("🏆 Best Model")
      st.write(best_model)
