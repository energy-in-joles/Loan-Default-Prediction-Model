import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from statsmodels.imputation import mice

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
	
train = pd.read_csv('Train-Data Loan.csv')
d = {'N':0,'Y':1}
train['Loan_Status'] = train['Loan_Status'].map(d)
train = clean(train)
features = list(train.columns[1:12])
target = np.array(train['Loan_Status'])

data = []
for index, row in train.iterrows():
	feat = list(row[features])
	data.append(feat)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100,min_samples_leaf=20)
clf = clf.fit(X_train, y_train)

scores = cross_val_score(clf,data,target,cv=10)
print(scores)
print(scores.mean())
