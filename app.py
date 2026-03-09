from flask import Flask, request, jsonify, render_template
from rag_core import ask_rag
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()
    message = data.get("message")

    result = ask_rag(message)

    print(result)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)