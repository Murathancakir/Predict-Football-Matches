from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

encoder = OneHotEncoder()

# load the model from disk
matchFHResultModel = pickle.load(open("AlgorithmFHResult.sav", 'rb'))
matchFTResultModel = pickle.load(open("AlgorithmFTResult.sav", 'rb'))
matchHighLowModel = pickle.load(open("AlgorithmHighLow.sav", 'rb'))

# read inference csv, excel
inferenceInput = pd.read_excel("C:\Workspace\PredictFootballMatches\PredictSuperLeague\csv\inferenceInputToday.xlsx")
accuracyInput = pd.read_excel("C:\Workspace\PredictFootballMatches\PredictSuperLeague\csv\inferenceInput.xlsx")
columns = ["HomeTeam","AwayTeam","Week"]


def encode(inferenceInput):
    encoded_data = encoder.fit_transform(inferenceInput[["HomeTeam","AwayTeam","Week"]])
    encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(columns))
    encodedInferenceInput = pd.concat([inferenceInput.drop(columns=columns), encoded_df], axis=1)
    
    for feature in matchHighLowModel.feature_names_in_:
        if feature not in inferenceInput:
            encodedInferenceInput[feature] = 0
    
    return encodedInferenceInput

encodedInferenceInput = encode(inferenceInput)
encodedAccuracyInput = encode(accuracyInput)

matchFHPrediction = matchFHResultModel.predict(encodedInferenceInput[matchFHResultModel.feature_names_in_])
# matchFHPredictionAll = matchFHResultModel.predict(encodedAccuracyInput[matchFHResultModel.feature_names_in_])
# FHResultAccuracy = (accuracy_score(encodedAccuracyInput["FHResult"], matchFHPredictionAll))

matchFTPrediction = matchFTResultModel.predict(encodedInferenceInput[matchFTResultModel.feature_names_in_])
# matchFTPredictionAll = matchFTResultModel.predict(encodedAccuracyInput[matchFTResultModel.feature_names_in_])
# FTResultAccuracy = (accuracy_score(encodedAccuracyInput["FTResult"], matchFTPredictionAll))

matchHighLowPrediction = matchHighLowModel.predict(encodedInferenceInput[matchHighLowModel.feature_names_in_])
# matchHighLowPredictionAll = matchHighLowModel.predict(encodedAccuracyInput[matchHighLowModel.feature_names_in_])
# highLowAccuracy = (accuracy_score(encodedAccuracyInput["HighLow"], matchHighLowPredictionAll))

x = 0
for i in inferenceInput[["HomeTeam","AwayTeam"]].values:
    print(f"{i[0]} - {i[1]}\n IY: {matchFHPrediction[x]} MS: {matchFTPrediction[x]} U/A: {matchHighLowPrediction[x]}")
    x += 1

# print(f"IY Overall Accuracy :{FHResultAccuracy}\nMS Overall Accuracy: {FTResultAccuracy}\nA/U Overall Accuracy: {highLowAccuracy}")

