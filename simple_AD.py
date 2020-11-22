import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
import pickle

joined_features = pd.read_csv("./dataset/joined_features.txt")

print(joined_features.head())

model = IsolationForest(n_estimators=200, 
                        max_samples=len(joined_features),
                        random_state=42,
                        n_jobs=-1,
                        verbose=2,
                        contamination=0.05)

predictions = model.fit_predict(joined_features)
predictions_str = ['Y' if i == -1 else 'N' for i in predictions] 
joined_features["ISFRAUD"] = predictions_str

joined_features.to_csv("./dataset/ftxn2_predicted.txt", index=False)


with open('isolation_forest_model','wb') as f:
    pickle.dump(model,f)

