from flask import Flask, jsonify, request
import sqlalchemy
from dotenv import load_dotenv
import xgboost as xgb
import json
import numpy as np

crc_model_path = './model/crc.model'

crc_model = xgb.Booster({'nthread': 4})
crc_model.load_model(crc_model_path)

app = Flask(__name__)

@app.route("/")
def main():
    return("Test")

@app.route('/compute-individual-risk', methods=['POST'])
def compute_individual_risk():
    data = request.get_json()
    hematocrit = data['hematocrit']
    hemoglobin = data['hemoglobin']
    platelets = data['platelets']
    wbc = data['wbc']
    sex = data['sex']
    age = data['age']

    if isinstance(hematocrit, list):
        model_data = xgb.DMatrix(np.stack((hematocrit, hemoglobin, platelets, wbc, sex, age)).T)
    elif isinstance(hematocrit, (int, float)):
        model_data = xgb.DMatrix(np.array([hematocrit, hemoglobin, platelets, wbc, sex, age]).reshape((1,6)))
    result = crc_model.predict(model_data).tolist()
    return(jsonify(result))

if __name__ == "__main__":
    app.run("0.0.0.0", port = 5000, debug = True)