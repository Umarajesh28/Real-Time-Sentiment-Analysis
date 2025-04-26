# Real-Time Sentiment Analysis for Customer Feedback

This project is an end-to-end web application that performs real-time sentiment analysis on customer feedback using a Neural Network (NN) model and Streamlit.

## Project Overview

- **Input:** Customer feedback (text reviews, comments).
- **Output:** Sentiment Prediction - Positive, Neutral, or Negative.
- **Backend Model:** Pre-trained BERT model (`cardiffnlp/twitter-roberta-base-sentiment`).
- **Frontend:** Streamlit web app for user interaction.
- **Deployment:** Ready for Streamlit Community Cloud or AWS deployment.

---

## Technologies Used

- Python
- Streamlit
- Hugging Face Transformers
- PyTorch
- Plotly (for visualization)

---

## Features

- Text input box for users to submit feedback.
- "Predict" button to trigger real-time sentiment prediction.
- "Clear" button to reset input.
- Pie chart visualization showing confidence distribution.
- Real-time output displaying Sentiment and Confidence Score.
- Ethical handling of user data (no storage of inputs).

---

##  How to Run Locally

1. Clone the repository:

``bash
   git clone https://github.com/Umarajesh28/Real-Time-Sentiment-Analysis-Project.git
   cd Real-Time-Sentiment-Analysis-Project/2_App/
