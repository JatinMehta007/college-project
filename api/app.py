import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle

app = Flask(__name__)

# Load models
claim_model = pickle.load(open("model-p.pkl", "rb"))
expense_model = pickle.load(open("data/model/trained_model.pkl", "rb"))

# ----------- Insurance Claim Logic -----------
def predict_claim(input_list):
    arr = np.array(input_list).reshape(1, -1)
    result = claim_model.predict(arr)
    return result[0]

@app.route('/')
def home():
    return render_template("home.html")  # default route

@app.route('/claim', methods=['GET', 'POST'])
def claim_prediction():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        values = list(to_predict_list.values())
        processed = []
        for x in values:
            if x.lower() == 'male':
                processed.append(1)
            elif x.lower() == 'female':
                processed.append(0)
            elif x.lower() == 'yes':
                processed.append(1)
            elif x.lower() == 'no':
                processed.append(0)
            else:
                processed.append(x)
        processed = list(map(float, processed))
        result = predict_claim(processed)
        prediction = 'Yes, the person can claim insurance.' if int(result) == 1 else 'No, the person cannot claim insurance.'
        return render_template("result_claim.html", prediction=prediction)

    return render_template("index_claim.html")


# ----------- Healthcare Expense Logic -----------
@app.route('/expenses', methods=['GET', 'POST'])
def expense_prediction():
    if request.method == 'POST':
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = expense_model.predict(final_features)
        output = round(prediction[0], 2)
        # Redirect to result page with query parameter
        return redirect(url_for('expense_result', result=output))

    return render_template("index_expense.html")

@app.route('/expense_result')
def expense_result():
    result = request.args.get('result', None)
    return render_template("result_expense.html", result=result)


# ----------- JSON API route ----------
@app.route('/results', methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = expense_model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
