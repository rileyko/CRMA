import pandas as pd
def read_dataset(file_path):
    raw_df = pd.read_csv('../Datasets/Opportunities.csv', skipinitialspace=True)
    digested_dataset = raw_df.dropna()
    OpptySources = digested_dataset['Opportunity_Source__c'].unique()
    OpptySourceDict = {string : i+1 for i,string in enumerate(OpptySources)}
    digested_dataset = digested_dataset.replace({'Origin': OpptySourceDict})
    digested_dataset = pd.get_dummies(digested_dataset, prefix='', prefix_sep='')
    for so in OpptySources:
        if so not in digested_dataset.columns:
            digested_dataset.loc[:,so] = 0
    digested_dataset = digested_dataset.reindex(sorted(digested_dataset.columns), axis=1)
    return digested_dataset