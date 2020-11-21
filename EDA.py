import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


faccount_df = pd.read_csv("./dataset/faccount.txt")
ftxn2_df = pd.read_csv("./dataset/ftxn2.txt")

print(ftxn2_df[ftxn2_df["ISFRAUD"] == 'N'])

