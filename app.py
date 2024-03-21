from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import pymongo
from pymongo import MongoClient

client = MongoClient("mongodb+srv://riza:989@testmd.pjvl08a.mongodb.net/test")
db = client['parks']
collection = db['results']

app = Flask(__name__)

@app.route('/', methods=['GET'])
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            mdvp_fo = float(request.form['mdvp_fo'])
            mdvp_shim = float(request.form['mdvp_shim'])
            nhr = float(request.form['nhr'])
            hnr = float(request.form['hnr'])
            rpde = float(request.form['rpde'])
            dfa = float(request.form['dfa'])
            spread1 = float(request.form['spread1'])
            ppe = float(request.form['ppe'])

            # loading the model file from the storage
            filename = 'model_knn3.sav'
            loaded_model = pickle.load(open(filename, 'rb'))

            # loading the scaler file from the storage
            scaler = pickle.load(open('standardscaler.sav', 'rb'))

            # scaling the input features
            features = scaler.transform([[mdvp_fo, mdvp_shim, nhr, hnr, rpde, dfa, spread1, ppe]])

            # predictions using the loaded model file
            prediction = loaded_model.predict(features)

            if prediction[0] == 1: 
                pred_knn3 = "You have Parkinson's Disease. Please consult a specialist."
            else:
                pred_knn3 = "You are a healthy person."
            print("Predicted Value:", prediction)

            # Store the prediction in the MongoDB collection
            post = {
                "mdvp_fo": mdvp_fo,
                "mdvp_shim": mdvp_shim,
                "nhr": nhr,
                "hnr": hnr,
                "rpde": rpde,
                "dfa": dfa,
                "spread1": spread1,
                "ppe": ppe,
                "prediction": pred_knn3
            }
            collection.insert_one(post)

            return render_template('results.html', prediction=pred_knn3)
        except Exception as e:
            print('The Exception message is: ', e)
            return 'Something went wrong'
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

