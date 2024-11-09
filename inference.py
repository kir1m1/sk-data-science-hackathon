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
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def read_data(path):
    df = pd.read_csv(path,low_memory=False)
    return df
    

def load_model(model_path):
    model = pickle.load(open(model_path,"rb"))
    return model
    
    
def preprocessing(df):
    '''Write your code to preprocess the dataframe to generate your features to be passed to the model and return the preprocessed dataframe'''   
    x_features = df.drop(['target'], axis=1)
    y_features = df['target']
    
    # split train 70% data & 30% test data
    X_train, X_test, y_train, y_test = train_test_split(x_features, y_features, test_size=0.3, random_state=42)

    unique_cat_feat_1_cols = x_features['cat_feature_1'].unique()
    unique_cat_feat_2_cols = x_features['cat_feature_2'].unique()
    scale_cols = [
    'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
    'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10',
    'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_15']
    
    # nominal_pipeline = Pipeline(steps = [
    #     ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    # ])
    cat_feat_1_pipeline = Pipeline(steps = [
        ('cat-feat-1-encode', OrdinalEncoder(categories=[unique_cat_feat_1_cols])),
    ])
    cat_feat_2_pipeline = Pipeline(steps = [
        ('cat-feat-2-encode', OrdinalEncoder(categories=[unique_cat_feat_2_cols])),
    ])
    col_transformer = ColumnTransformer(transformers = [
            ('cat-feat-1-pipeline', cat_feat_1_pipeline, ['cat_feature_1']),
            ('cat-feat-2-pipeline', cat_feat_2_pipeline, ['cat_feature_2']),
            ('scale', StandardScaler(), scale_cols),
        ],
        remainder = 'drop',
        n_jobs = -1
    )

    lin_reg = LinearRegression()
    final_pipeline = make_pipeline(col_transformer, lin_reg)
    final_pipeline.fit(X_train, y_train)

    cat_feat_1_encode = OrdinalEncoder(categories=[unique_cat_feat_1_cols])
    cat_feat_1_transform = cat_feat_1_encode.fit_transform(X_test[['cat_feature_1']])
    
    cat_feat_2_encode = OrdinalEncoder(categories=[unique_cat_feat_2_cols])
    cat_feat_2_transform = cat_feat_2_encode.fit_transform(X_test[['cat_feature_2']])
    
    X_test[['cat_feature_1']] = cat_feat_1_transform
    X_test[['cat_feature_2']] = cat_feat_2_transform
    df_processed = pd.concat([X_test], axis=1)
    df_processed['target'] = y_test
    
    print(df_processed.columns)
    print(df_processed)
    
    return df_processed
    

def inference(df_processed, model):
    '''Write your code to pass the preprocessed dataframe to your model and generate predictions from the model and return the predictions'''
    
    preds = model.predict(df_processed)

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
    
    

    
    
