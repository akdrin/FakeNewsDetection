from flask import Flask, request, render_template
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

MODEL_PATH = "xlmr_multiclass"

if not os.path.exists(MODEL_PATH):
    raise Exception(f"Model folder '{MODEL_PATH}' not found. Check your path.")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
except Exception as e:
    raise Exception(f"Error loading model: {e}")

model.eval()

DISPLAY_LABELS = {
    "LABEL_0": "Fake News",
    "LABEL_1": "Real News",
    "LABEL_2": "Uncertain",
}


@app.route('/', methods=['GET', 'POST'])
def predict():
    result = ""
    confidence = ""

    if request.method == 'POST':
        text = request.form.get("news")

        if not text or text.strip() == "":
            result = "Please enter some text"
            return render_template("home.html", result=result)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()

        raw_label = model.config.id2label.get(pred, f"LABEL_{pred}")
        result = DISPLAY_LABELS.get(raw_label, raw_label)
        confidence = f"{round(conf * 100, 2)}%"

    return render_template("home.html", result=result, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)
