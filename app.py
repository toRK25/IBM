import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vector = pickle.load(open("vector.pkl", "rb"))

st.set_page_config(page_title="Kiko Dashboard", layout="wide")

# Sidebar
st.sidebar.title("Kiko Controls")
st.sidebar.write("Live Chat Filter")

# Main Title
st.title("📊 Kiko Dashboard")

# Input section
st.subheader("💬 Enter Chat Message")
user_input = st.text_input("Type message here:")

def predict_text(text):
    vec = vector.transform([text])
    return model.predict(vec)[0]

# Prediction
if user_input:
    result = predict_text(user_input)
    
    st.subheader("🔍 Prediction Result")
    
    if result == "negative":
        st.error("🚫 Negative (Blocked)")
    elif result == "positive":
        st.success("✅ Positive (Allowed)")
    else:
        st.warning("😐 Neutral")

# Divider
st.markdown("---")

# Dummy stats (you can replace later)
st.subheader("📈 Model Insights")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", "88.7%")
col2.metric("Model", "LinearSVC")
col3.metric("Features", "TF-IDF")

# Sample chart
import pandas as pd
data = pd.DataFrame({
    "Class": ["Positive", "Negative", "Neutral"],
    "Count": [50, 30, 20]
})

st.bar_chart(data.set_index("Class"))