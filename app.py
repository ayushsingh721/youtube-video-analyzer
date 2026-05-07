from flask import Flask, render_template, request, jsonify
from nlp_engine import preprocess_text
from question_generator import generate_questions

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()

    if not data or "notes" not in data:
        return jsonify({"error": "No notes provided"}), 400

    notes = data["notes"].strip()

    if not notes:
        return jsonify({"error": "Notes are empty"}), 400

    # NLP preprocessing
    processed = preprocess_text(notes)

    # Generate questions
    result = generate_questions(processed)

    return jsonify(result)
if __name__ == "__main__":
    app.run(debug=True)
