from importing_lib import *
from data_ingestion import df        # ← import df from data_ingestion

scaleIt = MinMaxScaler()
columns_to_be_scaled = [c for c in df.columns if df[c].max() > 1]
print("The columns which are to be scaled are :",columns_to_be_scaled)

scaled_columns = scaleIt.fit_transform(df[columns_to_be_scaled])
scaled_columns = pd.DataFrame(scaled_columns, columns=columns_to_be_scaled)
scaled_columns['Outcome'] = df['Outcome']