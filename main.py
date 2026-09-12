from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

with open("model_treated.pkl", "rb") as f:
    model_treated_conf = pickle.load(f)

with open("model_control.pkl", "rb") as f:
    model_control_conf = pickle.load(f)

with open("q_combined.pkl", "rb") as f:
    q_combined = pickle.load(f)

class UserFeatures(BaseModel):
    age: int
    account_balance: float
    activity_score: float

@app.post("/score_uplift")
def score_uplift(user: UserFeatures):
    features = pd.DataFrame([[user.age, user.account_balance, user.activity_score]],
                             columns=["age", "account_balance", "activity_score"])

    pred_treated = model_treated_conf.predict(features)[0]
    pred_control = model_control_conf.predict(features)[0]
    effect = pred_treated - pred_control

    # same conformal margin from phase 3, just applied live instead of on a dataframe
    lower = effect - q_combined
    upper = effect + q_combined

    return {
        "predicted_effect": round(float(effect), 2),
        "lower_bound": round(float(lower), 2),
        "upper_bound": round(float(upper), 2),
        "recommend_treat": bool(lower > 0)
    }