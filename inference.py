import numpy as np
import pandas as pd
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
import argparse
import os
import pickle
import json
from sklearn.model_selection import train_test_split

def read_data(path):
    df = pd.read_csv(path,low_memory=False)
    return df
    

def load_model(model_path):
    model = pickle.load(open(model_path,"rb"))
    return model
    
    
def preprocessing(df):
    '''Write your code to preprocess the dataframe to generate your features to be passed to the model and return the preprocessed dataframe'''
    df_processed = df

    return df_processed
    

def inference(df_processed, model):
    '''Write your code to pass the preprocessed dataframe to your model and generate predictions from the model and return the predictions'''
    lin_reg_model = LinearRegression()
    y = df_processed['target'].values
    feat_cols = df_processed.drop(['target', 'cat_feature_1', 'cat_feature_2', axis=1)
    preds = dict()
    
    for col in feat_cols.columns:
        x = feat_cols.loc[:, col].values.reshape(-1, 1)
        lin_reg_model.fit(x, y)
        preds[col] = lin_reg_model.predict(x)

    return preds
    
    
    
def main():
    cur_dir = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',type=str,default=cur_dir)
    # parser.add_argument('--data_dir',type=str,default='data')
    # parser.add_argument('--model_dir',type=str,default='models')
    parser.add_argument('--dataset',type=str,default='case_study_validation_data.csv')
    parser.add_argument('--model_name',type=str,default='model.pkl')
    parser.add_argument('--output_dir',type=str,default='output')
    parser.add_argument('--output_name',type=str,default="inference_results.json")
    args = parser.parse_args()
    
    cur_dir = args.base_dir
    # data_dir = os.path.join(cur_dir,args.data_dir)
    # model_dir = os.path.join(cur_dir,args.model_dir)
    # dataset_path = os.path.join(data_dir,args.dataset_name)
    # model_path = os.path.join(model_dir,args.model_name)
    dataset_path = os.path.join(cur_dir,args.dataset)
    model_path = os.path.join(cur_dir,args.model_name)
    
    df = read_data(dataset_path)
    model = load_model(model_path)
    
    df_processed = preprocessing(df)
    
    predictions = inference(df_processed[[col for col in df_processed.columns if col != 'target']], model)
    
    print("The Root Mean Squared Error for the model is {}".format(root_mean_squared_error(df_processed['target'], predictions)))
    
    results = {}
    results['rmse'] = root_mean_squared_error(df_processed['target'], predictions)
    output_path = os.path.join(args.base_dir, args.output_dir, args.output_name)
    with open(output_path, "w") as outfile: 
        json.dump(results, outfile)
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    

    
