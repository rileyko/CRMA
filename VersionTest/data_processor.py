import pandas as pd
import numpy as np

LABEL_COLUMN = 'Opportunity_Source__c'
LABEL_COLUMNS = ['AE','Alliances','BDR','Marketing']
COLUMN_NAMES = ['Amount','ExpectedRevenue','TotalOpportunityQuantity']


def read_dataset(file_path):
    raw_df = pd.read_csv(file_path, skipinitialspace=True)
    digested_dataset = raw_df.dropna()
    OpptySources = digested_dataset[LABEL_COLUMN].unique()
    OpptySourceDict = {string : i+1 for i,string in enumerate(OpptySources)}
    digested_dataset = digested_dataset.replace({'Origin': OpptySourceDict})
    digested_dataset = pd.get_dummies(digested_dataset, prefix='', prefix_sep='')
    for so in OpptySources:
        if so not in digested_dataset.columns:
            digested_dataset.loc[:,so] = 0
    digested_dataset = digested_dataset.reindex(sorted(digested_dataset.columns), axis=1)
    return digested_dataset

def preprocessor(file_path):
    dataset = read_dataset(file_path)
    dataset.drop(columns=LABEL_COLUMNS, inplace=True, errors='ignore')
    return dataset

def postprocessor(predictions):
    results = []
    for prediction in predictions: # predictions is an array of array of strings
        results.append(
            {
            "AE":np.round_(prediction[0].astype(np.float64),3),
            "Alliances":np.round_(prediction[1].astype(np.float64),3),
            "BDR":np.round_(prediction[2].astype(np.float64),3),
            "Marketing":np.round_(prediction[3].astype(np.float64),3)
            })
    return results
