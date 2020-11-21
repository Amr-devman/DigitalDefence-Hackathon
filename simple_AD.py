import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

faccount_df = pd.read_csv("./dataset/faccount.txt")
faccount_df['FULLNAME'] = faccount_df['FIRSTNAME']+" "+faccount_df['SURNAME']


ftxn2_df = pd.read_csv("./dataset/ftxn2.txt")
ftxn2_df = ftxn2_df.drop(columns=[
                                    "TXN_ID",
                                    "ISFRAUD",
                                    "ISFLAGGED",
                                    "REFERENCE",
                                    "TXTYPE",
                                    "TXDATE"]
                        )

def id_to_name(id,id_name_dict):
    return id_name_dict["FULLNAME"][int(id)]



id_and_names = faccount_df[["ACCTID","FULLNAME"]].set_index("ACCTID").to_dict()


# ftxn2_df["FROMNAME"] = np.vectorize(id_to_name)(ftxn2_df["FROMACCTID"], id_and_names)
# ftxn2_df["TONAME"] = np.vectorize(id_to_name)(ftxn2_df["TOACCTID"], id_and_names)


print(ftxn2_df.columns)
model = IsolationForest(n_estimators=100, 
                        max_samples=len(ftxn2_df),
                        random_state=42,
                        n_jobs=-1,
                        verbose=1)

predictions = model.fit_predict(ftxn2_df)
predictions_str = ['Y' if i == -1 else 'N' for i in predictions] 
ftxn2_df["ISFRAUD"] = predictions_str

ftxn2_df.to_csv("./dataset/ftxn2_predicted.txt", index=False)



