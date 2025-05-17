import streamlit as st 
import plotly.express as px
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re

# Title and subtitle
st.title("Real-Time Sentiment Analysis App")
st.markdown("Analyze customer feedback instantly using a pre-trained BERT model (RoBERTa).")

st.markdown("---")
st.markdown("_Enter any feedback, review, or tweet to get an instant sentiment prediction._")

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
positive_threshold = 0.6  # Updated threshold for Positive sentiment
negative_threshold = 0.6  # Updated threshold for Negative sentiment

# Additional Preprocessing Function (to handle common phrases)
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Handle common phrases or rephrase negative statements
    text = re.sub(r'\b(not bad|could be better|not great|kind of good)\b', 'neutral', text)
    
    # Safely handle negation phrases like "not good", "not happy"
    def handle_negation(match):
        words = match.group().split()
        if len(words) >= 2:
            return 'not_' + words[1]
        else:
            return match.group()  # fallback to original if structure unexpected
    
    text = re.sub(r'\bnot \w+', handle_negation, text)

    return text

# Initialize session state for user input
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Setup clear flag in session state
if "clear_text" not in st.session_state:
    st.session_state.clear_text = False

# Clear user_input before the widget is rendered if flagged
if st.session_state.clear_text:
    st.session_state.user_input = ""
    st.session_state.clear_text = False    

# User input
user_input = st.text_area("Enter customer feedback here:", key="user_input", height=150)


col1, col2 = st.columns([1, 1])

with col1:
    predict_button = st.button("Predict Sentiment", use_container_width=True)

with col2:
    clear_button = st.button("Clear", use_container_width=True)

if clear_button:
    st.session_state.clear_text = True
    st.rerun()  # Refresh the app to clear the text area and reset the UI

if predict_button:
    if user_input.strip() != "":
        # Preprocess text before prediction
        processed_input = preprocess_text(user_input)
        result = sentiment_pipeline(processed_input)
        
        # Get predicted label and score
        label = labels[result[0]['label']]
        score = result[0]['score']
        
        # Apply stricter thresholds for classification
        if score < positive_threshold and label == "Positive":
            label = "Neutral"  # If Positive score is too low, classify as Neutral
        elif score < negative_threshold and label == "Negative":
            label = "Neutral"  # If Negative score is too low, classify as Neutral
        
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

