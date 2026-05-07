import os
from flask import Flask, render_template, request, jsonify
from analyzer import VideoAnalyzer
from utils import extract_video_id

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Add your YouTube API key here
YOUTUBE_API_KEY = "PASTE_YOUR_NEW_API_KEY_HERE"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_video():
    try:
        data = request.get_json()
        user_input = data.get("url", "").strip()

        # Check if input is empty
        if user_input == "":
            return jsonify({
                "error": "Please enter a YouTube URL or Video ID."
            }), 400

        # Get video ID from URL/input
        video_id = extract_video_id(user_input)

        if video_id is None:
            return jsonify({
                "error": "Invalid YouTube URL or Video ID."
            }), 400

        # Check API key
        if YOUTUBE_API_KEY == "PASTE_YOUR_NEW_API_KEY_HERE":
            return jsonify({
                "error": "Please add your YouTube API key in app.py"
            }), 400

        # Analyze video
        analyzer = VideoAnalyzer(YOUTUBE_API_KEY)
        result = analyzer.analyze(video_id)

        # If analyzer returns error
        if "error" in result:
            return jsonify(result), 400

        return jsonify(result)

    except Exception as err:
        return jsonify({
            "error": "Server error: " + str(err)
        }), 500

if __name__ == "__main__":
    print("\nStarting YouTube Video Analyzer...")
    
    if YOUTUBE_API_KEY != "PASTE_YOUR_NEW_API_KEY_HERE":
        print("API Key Loaded Successfully")
    else:
        print("API Key Missing")

    print("Server running on http://127.0.0.1:5000\n")

    app.run(debug=True)
