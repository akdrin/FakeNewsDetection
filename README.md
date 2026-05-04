# Fake News Detection using XLM-RoBERTa

This project is a web application that detects whether a news article is **Real**, **Fake**, or **Uncertain** using a deep learning model.

---

## Overview

The system uses a fine-tuned XLM-RoBERTa model to analyze news text and classify it into three categories. A Flask backend is used to serve predictions through a simple web interface.

---

## Features

- Classifies news into Real, Fake, or Uncertain  
- Provides confidence score for predictions  
- Handles long text input  
- Simple web interface using Flask  

---

## Tech Stack

- Python  
- Flask  
- PyTorch  
- Hugging Face Transformers  
- HTML + Tailwind CSS  

---

## Project Structure

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
▶️ Run the Application
python app.py

Open in browser:
http://127.0.0.1:5000/
