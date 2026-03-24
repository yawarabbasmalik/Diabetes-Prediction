from importing_lib import *

#importing the data file

df = pd.read_csv("diabetes.csv", index_col=False)
print(df.head())

print ("************************")

print(df.info())

print ("************************")