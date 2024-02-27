## This script merge the results from the experiments folder and the experiments.csv file into merged_experiments.csv which contains the f1, em and precision values for each experiment.

import os
import pandas as pd
import json
# Read experiments.csv into a pandas dataframe
experiments_df = pd.read_csv('experiments.csv')

# Iterate over each row in the dataframe
for index, row in experiments_df.iterrows():
    # Get the UUID and folder path
    uuid = row['UUID']
    folder_path = f"experiments/{uuid}"

    # Read the squad_metric.json file into a pandas dataframe
    metric_file = os.path.join(folder_path, 'squad_metric.json')
    if not os.path.exists(metric_file):
        continue
    with open(metric_file, 'r') as f:
        metric_data = json.load(f)
    metric_df = pd.DataFrame.from_dict(metric_data, orient='columns')
    #print(metric_df)
    #exit()
    
    # Extract the "f1" values from the metric dataframe and add them as new columns to the experiments dataframe
    for col in metric_df.columns:
        if col == 'predictions':
            if 'f1' in metric_df[col]:
                experiments_df.loc[index, 'f1'] = metric_df[col]['f1']
            if 'exact_match' in metric_df[col]:
                experiments_df.loc[index, 'em'] = metric_df[col]['exact_match']
            if 'precision' in metric_df[col]:
                experiments_df.loc[index, 'precision'] = metric_df[col]['precision']                
        else:
            if 'f1' in metric_df[col]:
                experiments_df.loc[index, 'f1_correct'] = metric_df[col]['f1']
            if 'exact_match' in metric_df[col]:
                experiments_df.loc[index, 'em_correct'] = metric_df[col]['exact_match']            
            if 'precision' in metric_df[col]:
                experiments_df.loc[index, 'precision_correct'] = metric_df[col]['precision']                        
# Print the updated experiments dataframe
# Set pandas to display all columns
experiments_df = experiments_df[experiments_df['f1_correct'].notnull() | experiments_df['f1'].notnull()] # ignore experiments that did not run (error) or aborted.
experiments_df.to_csv('merged_experiments.csv', index=False)