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

##  Demo Screenshots

### 1. Home Page - Feedback Input and Sentiment Prediction
![App Main Page](./2_App/app_main_page.png)

### 2. Sentiment Confidence Pie Chart
![Sentiment Confidence Pie Chart](./2_App/app_pie_chart.png)


---
##  How to Run Locally

1. Clone the repository:

``bash
   git clone https://github.com/Umarajesh28/Real-Time-Sentiment-Analysis-Project.git
   cd Real-Time-Sentiment-Analysis-Project/2_App/

2. Install required libraries:
   
``bash
   pip install -r requirements.txt

3. Run the Streamlit app:

``bash
   streamlit run app.py

---

##  Deployment


You can deploy this app easily on [Streamlit Community Cloud](https://streamlit.io/cloud) or any cloud platform like AWS EC2.

### AWS EC2 Deployment Steps

1. Launch an EC2 instance   
2. Connect to EC2 instance via SSH    
3. Update and install dependencies
4. Clone your project repository
5. Create and activate a Python virtual environment
6. Install required Python packages
7. Run the Streamlit app
8. Access your app @ http://your-ec2-public-ip:8501
   

##  Extras

- As part of this project, I also explored training a custom neural network model on the **TweetEval Sentiment dataset**.
- This work is available in the `TweetEval_Sentiment_NeuralNet.ipynb` notebook.
- However, for better performance, higher accuracy, and real-time prediction capability, I chose to deploy a **fine-tuned pre-trained BERT model** for the final Streamlit application.

---

##  Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Cardiff NLP Twitter RoBERTa Sentiment Model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
- [Streamlit](https://streamlit.io/)

---
## Live Demo

You can view the live demo of this app [here](https://nxedwjbpeobwcfbyexavyp.streamlit.app/).



