import streamlit as st
import plotly.express as px
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re

# Title and subtitle
st.title("Real-Time Sentiment Analysis App")
st.markdown("Analyze customer feedback instantly using a pre-trained BERT model (RoBERTa).")

st.markdown("---")
st.markdown("📢 _Enter any feedback, review, or tweet to get an instant sentiment prediction._")

# Load tokenizer and model
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiment_pipeline = load_model()

# Label mapping (including custom sentiment categories)
labels = {
    'LABEL_0': 'Negative',
    'LABEL_1': 'Neutral',
    'LABEL_2': 'Positive'
}

# Custom sentiment categories for clearer classification
custom_labels = {
    'Positive': ['Excellent', 'Good', 'Great', 'Awesome'],
    'Negative': ['Bad', 'Terrible', 'Horrible', 'Worst'],
    'Neutral': ['Okay', 'Fine', 'Average', 'Neutral']
}

# Confidence thresholds for classification
positive_threshold = 0.7
negative_threshold = 0.7

# Additional Preprocessing Function (to handle common phrases)
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Handle common phrases or rephrase negative statements
    text = re.sub(r'\b(not bad|could be better|not great|kind of good)\b', 'neutral', text)
    
    # Handling negations (simple)
    text = re.sub(r'\b(not )+\w+', 'not_' + text.split()[1], text)
    
    # More preprocessing based on your dataset
    return text

# Initialize session state for user input
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# User input
user_input = st.text_area("Enter customer feedback here:", height=150, value=st.session_state.user_input)

col1, col2 = st.columns([1, 1])

with col1:
    predict_button = st.button("Predict Sentiment", use_container_width=True)

with col2:
    clear_button = st.button("Clear", use_container_width=True)

if clear_button:
    st.session_state.user_input = ""  # Reset the text area input

if predict_button:
    if user_input.strip() != "":
        # Preprocess text before prediction
        processed_input = preprocess_text(user_input)
        result = sentiment_pipeline(processed_input)
        
        # Get predicted label and score
        label = labels[result[0]['label']]
        score = result[0]['score']
        
        # Apply thresholds for more deterministic sentiment classification
        if label == "Neutral" and score < 0.6:
            if result[0]['label'] == 'LABEL_2':  # Positive sentiment
                label = "Positive"
                score = result[0]['score']
            elif result[0]['label'] == 'LABEL_0':  # Negative sentiment
                label = "Negative"
                score = result[0]['score']
        
        # Check if sentiment matches any of the custom categories
        sentiment_category = ""
        for category, keywords in custom_labels.items():
            if any(keyword.lower() in processed_input for keyword in keywords):
                sentiment_category = category
                break
        
        if sentiment_category:
            st.success(f"**Sentiment Category:** {sentiment_category}  \n**Confidence:** {score:.2f}")
        else:
            st.success(f"**Sentiment:** {label}  \n**Confidence:** {score:.2f}")
        
        # Plot pie chart for sentiment confidence distribution
        df = pd.DataFrame({'Sentiment': [label, 'Other'], 'Confidence': [score, 1 - score]})
        fig = px.pie(df, names='Sentiment', values='Confidence', title='Sentiment Confidence Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("Please enter some text before clicking Predict.")
