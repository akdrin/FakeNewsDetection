📰 Fake News Detection using XLM-RoBERTa

An AI-powered web application that classifies news articles as Real, Fake, or Uncertain using a transformer-based deep learning model.

🚀 Overview

This project uses a fine-tuned XLM-RoBERTa model to analyze textual news content and predict its authenticity. The system is deployed with a Flask backend and a clean web interface for real-time predictions.

✨ Features
🔍 Classifies news into Real / Fake / Uncertain
⚡ Real-time predictions via Flask web app
📊 Displays confidence scores
🌍 Uses multilingual transformer (XLM-R)
🧠 Handles long text inputs (up to 512 tokens)
🧠 Model Details
Model: XLM-RoBERTa (xlm-roberta-base)
Task: Multiclass Text Classification
Framework: PyTorch
Tokenization: Byte-Pair Encoding (BPE)
Output: Class label + probability score
🛠️ Tech Stack
Backend: Flask
ML Framework: PyTorch
NLP Library: Transformers (Hugging Face)
Frontend: HTML + Tailwind CSS
Model Format: .safetensors
📂 Project Structure
project/
│── app.py
│── requirements.txt
│── xlmr_multiclass/
│     ├── config.json
│     ├── model.safetensors
│     ├── tokenizer.json
│     ├── tokenizer_config.json
│     ├── vocab files...
│── templates/
│     └── home.html
⚙️ Installation
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
▶️ Run the Application
python app.py

Open in browser:

http://127.0.0.1:5000/
