import numpy as np
import pandas as pd
from sklearn import tree

train = pd.read_csv('Train-Data Loan.csv') #replace csv file with training data
applications = pd.read_csv('Test-Data Loan.csv') #replace csv file with data to predict

def clean(df):
    d = {'Rural':0,'Semiurban':1,'Urban':2}
    df['Property_Area'] = df['Property_Area'].map(d)
    d = {'No':0,'Yes':1}
    df['Self_Employed'] = df['Self_Employed'].map(d)
    d = {'Not Graduate':0,'Graduate':1}
    df['Education'] = df['Education'].map(d)
    d = {'0':0,'1':1,'2':2,'3+':3}
    df['Dependents'] = df['Dependents'].map(d)
    d = {'No':0,'Yes':1}
    df['Married'] = df['Married'].map(d)
    d = {'Male':0,'Female':1}
    df['Gender']= df['Gender'].map(d)
    df = df.fillna(df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']].median())
    df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
    return df

d = {'N':0,'Y':1}
train['Loan_Status'] = train['Loan_Status'].map(d)
train = clean(train)
features = list(train.columns[1:12])
y = train['Loan_Status']
x = train[features]
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100,min_samples_leaf=20)
clf = clf.fit(x, y)

applications_prediction = applications.copy()
applications = clean(applications)
applications_prediction['Loan_Status'] = np.nan
applications_prediction.drop(features,axis=1,inplace=True)

for index, row in applications.iterrows():
	applications_data = row[features].tolist()
	num_ans = clf.predict([applications_data])
	ans = [key for key, value in d.items() if value == num_ans[0]][0]
	applications_prediction.loc[index,'Loan_Status'] = ans
applications_prediction.to_csv('Loan Prediction.csv',index=False)
