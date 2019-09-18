#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 09:24:09 2018
@author: nealbayya
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def clean(df):
    df = df.drop(columns=['Name','PassengerId','Cabin'])
    sex = pd.get_dummies(df.loc[:,'Sex']) #add one hot encode sex and drop gender var
    df = pd.concat([df,sex], axis = 1)
    df = df.drop(columns=['Sex'])
    
    embarked = pd.get_dummies(df.loc[:,'Embarked']) #same one hot procedure
    df = pd.concat([df,embarked], axis = 1)
    df = df.drop(columns=['Embarked'])
    
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['SibSp'].fillna(df['SibSp'].mean(), inplace=True)
    df['Parch'].fillna(df['Parch'].mean(), inplace=True)
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    
    ticket_col  = []
    ticket_repl = set()
    for idx, t in enumerate(df.loc[:,'Ticket'].values):
        try: ticket_col.append(int(str(t).split(' ')[-1]))
        except: 
            ticket_col.append(0)
            ticket_repl.add(idx)
    ticket_mean = sum(ticket_col)/(len(ticket_col)-len(ticket_repl))
    for idx in ticket_repl: ticket_col[idx] = ticket_mean
    
    df['Ticket'] = ticket_col
    
    return df

def output(testIds, predictY):
	with open('out.csv', 'w') as f:
	    f.write('PassengerId,Survived\n')
	    for i in range(len(predictY)):
	        f.write(str(testIds[i])+","+str(predictY[i])+"\n")

def main():
    trainpath = '~/.kaggle/competitions/titanic/train.csv'
    testpath = '~/.kaggle/competitions/titanic/test.csv'
    
    #clean and prepare data
    traindf = clean(pd.read_csv(trainpath))
    trainY = traindf['Survived']
    trainX = traindf.drop(columns =['Survived'])
    
    testX = pd.read_csv(testpath)
    testIds = testX['PassengerId']
    testX = clean(testX)
    
    random_forest = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
    random_forest.fit(trainX, trainY)
    
    predictY = random_forest.predict(testX)
    output(testIds, predictY)

if __name__ == '__main__':
    main()
    
    
    
    
    
    
