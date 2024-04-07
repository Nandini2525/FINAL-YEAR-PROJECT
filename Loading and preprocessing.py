import numpy as np
import pandas as pd

df=pd.read_csv("/content/SDN_Intrusion.csv")
df.dropna(axis='columns')
empty_cols = [col for col in df.columns if df[col].isnull().any()]
empty_cols
class_counts = df['Class'].value_counts()
class_counts_df = pd.DataFrame({'Class': class_counts.index, 'Count': class_counts.values})
print(class_counts_df)
df.Class = pd.Categorical(df.Class)
df['code'] = df.Class.cat.codes
df['code']

df.drop('Class', inplace=True, axis=1)
df2=df.dropna()

df2.drop('Unnamed: 0', inplace=True, axis=1)
df2 = df2[np.isfinite(df2).all(1)]

df2
df3=df2.sample(n=10000,replace=True)
df4=df3.drop('code', axis=1)
df4

X=df4
y=df3['code']
X= df2.iloc[: , :-1]
y= df2.iloc[:,-1:]
y
