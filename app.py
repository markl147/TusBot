from chatbot import generate_response
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form["user_input"]
    response = generate_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
