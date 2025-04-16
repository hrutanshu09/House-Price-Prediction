from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# ✅ Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# 🔹 Intro page
@app.route("/")
def intro():
    return render_template("intro.html")

# 🔹 Main prediction page (GET = show form, POST = show prediction)
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            crim = float(request.form['crim'])
            rm = float(request.form['rm'])
            age = float(request.form['age'])
            tax = float(request.form['tax'])
            lstat = float(request.form['lstat'])

            features = np.array([[crim, rm, age, tax, lstat]])
            prediction = model.predict(features)[0] * 20  # boost predicted price

            return render_template(
                "index.html",
                predicted=prediction,
                crim=crim,
                rm=rm,
                age=age,
                tax=tax,
                lstat=lstat
            )
        except Exception as e:
            return render_template("index.html", error=str(e), predicted=None)

    # On GET request
    return render_template("index.html", predicted=None)

if __name__ == "__main__":
    app.run(debug=True)
