from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Open pickle files in binary mode ('rb')
cv = pickle.load(open("models/cv.pkl", "rb"))
clf = pickle.load(open("models/clf.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form.get('content')
    tokenized_email = cv.transform([email])  # Transform the input email using the vectorizer
    prediction = clf.predict(tokenized_email)  # Predict using the classifier
    prediction = 1 if prediction == 1 else -1  # Convert prediction to 1 or -1
    return render_template("index.html", prediction=prediction, email=email)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
