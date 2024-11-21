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
from sklearn.impute import SimpleImputer

def read_data(path):
    df = pd.read_csv(path,low_memory=False)
    return df
    

def load_model(model_path):
    model = pickle.load(open(model_path,"rb"))
    return model
    
    
def preprocessing(df):
    '''Write your code to preprocess the dataframe to generate your features to be passed to the model and return the preprocessed dataframe''' 
    # separate features and target
    x_features = df.drop(['target'], axis=1)
    y_features = df['target']
    
    # split train 70% data & 30% test data
    X_train, X_test, y_train, y_test = train_test_split(x_features, y_features, test_size=0.3, random_state=42)

    # define categorical & continuous features
    cat_cols = ['cat_feature_1', 'cat_feature_2']
    scale_cols = [
        'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
        'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10',
        'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_15'
    ]

    # pipeline for categorical features
    cat_feat_pipeline = Pipeline(steps = [
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        # ('cat_encode', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ('cat-encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    # pipeline for continuous features
    cont_feat_pipeline = Pipeline(steps = [
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler()),
    ])

    # column transformer
    col_transformer = ColumnTransformer(transformers = [
            ('cat_encode', cat_feat_pipeline, cat_cols),
            ('scale',      cont_feat_pipeline, scale_cols),
        ],
        remainder="drop",
        n_jobs=-1,
        verbose_feature_names_out=False)

    # setup final pipeline
    final_pipeline = Pipeline(steps = [
        ('preprocessor', col_transformer),
        ('regressor', LinearRegression()),
    ])

    # fit training data
    final_pipeline.fit(X_train, y_train)


    # transform test dataset with the trained pipeline
    X_test_transformed = final_pipeline.named_steps['preprocessor'].transform(X_test)
    df_processed = pd.DataFrame(X_test_transformed, columns=col_transformer.get_feature_names_out())

    # Add target variable back to the processed test data
    df_processed['target'] = y_test.reset_index(drop=True)
        
    return df_processed

def preprocezz(df_encoded):
    # Read 1  the CSV file into a DataFrame
    df = pd.read_csv('case_study_validation_data.csv')
    
    # Convert all column names to strings
    df.columns = df.columns.astype(str)
    
    # Create a OneHotEncoder object
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Fit the encoder to the columns cat_feature_1 and cat_feature_2
    encoder.fit(df[['cat_feature_1', 'cat_feature_2']])
    
    # Transform the columns cat_feature_1 and cat_feature_2
    encoded_features = encoder.transform(df[['cat_feature_1', 'cat_feature_2']])
    
    # Get the feature names from the encoder
    encoded_feature_names = encoder.get_feature_names_out(['cat_feature_1', 'cat_feature_2'])
    
    # Create a new DataFrame by concatenating the existing DataFrame with the encoded features
    df_encoded = pd.concat([df, pd.DataFrame(encoded_features, columns=encoded_feature_names)], axis=1).drop(['cat_feature_1', 'cat_feature_2'], axis=1)
    
    # Split the data into training and testing sets
    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and fit a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return df_encoded
    

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
    
    # df_processed = preprocessing(df)
    df_processed = preprocezz(df)
    
    predictions = inference(df_processed[[col for col in df_processed.columns if col != 'target']], model)
    
    print("The Root Mean Squared Error for the model is {}".format(root_mean_squared_error(df_processed['target'], predictions)))
    
    results = {}
    results['rmse'] = root_mean_squared_error(df_processed['target'], predictions)
    output_path = os.path.join(args.base_dir, args.output_dir, args.output_name)
    with open(output_path, "w") as outfile: 
        json.dump(results, outfile)
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    

    
    
