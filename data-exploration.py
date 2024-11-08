import pandas as pd
import numpy as np
import json
import argparse
import os
#Import any other necessary modules
from sklearn.linear_model import Lasso


def load_data(path):
    df = pd.read_csv(path,low_memory=False)
    return df


def get_highest_correlated_feature(df):
    '''Implement your code to return the feature
    having the highest correlation with the 'target' variable'''
    col_names_to_drop = ["target", "cat_feature_1", "cat_feature_2"]
    X = df.drop(col_names_to_drop, axis=1).values
    y = df['target'].values
    ft_col_names = df.drop(col_names_to_drop, axis=1).columns
    coefficients = Lasso.fit(X,y).coef_
    max_coef_idx = np.argmax(np.abs(coefficients))
    
    return ft_col_names[max_coef_idx]



def category_with_highest_mean_cat_feature_2(df):
    '''Implement your code to return the category in 'cat_feature_2'
    variable which has the highest mean for the 'target' variable'''
    
    return



def abs_std_dev_diff_btwn_groups_cat_feature_1(df):
    '''Implement your code to return the absolute difference in standar deviation
    of the 'target' variable between the 2 groups present in 'cat_feature_1' '''
    
    return



def min_feature_8_for_cat_feature_2(df):
    '''Implement your code to return the minimum value of 'feature_8'
    for the group Category_C in the variable 'cat_feature_2' '''
    
    return



def get_variance_feature_12_for_group(df):
    '''Implement your code to return the variance of 'feature_12' for the 
    group where 'cat_feature_2' is Category_A and 'cat_feature_1' is High'''
    
    return




def main():
    cur_dir = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',type=str,default=cur_dir)
    parser.add_argument('--data_dir',type=str,default='')
    parser.add_argument('--dataset',type=str,default='case_study_data.csv')
    parser.add_argument('--output_dir',type=str,default='output')
    parser.add_argument('--output_name',type=str,default="calculated_metrics.json")
    args = parser.parse_args()

    data_path = os.path.join(cur_dir,args.data_dir,args.dataset)
    df = load_data(data_path)
    results = {}
    results['get_highest_correlated_feature'] = get_highest_correlated_feature(df)
    results['category_with_highest_mean_cat_feature_2'] = category_with_highest_mean_cat_feature_2(df)
    results['abs_std_dev_diff_btwn_groups_cat_feature_1'] = round(abs_std_dev_diff_btwn_groups_cat_feature_1(df),2)
    results['min_feature_8_for_cat_feature_2'] = round(min_feature_8_for_cat_feature_2(df),2)
    results['get_variance_feature_12_for_group'] = round(get_variance_feature_12_for_group(df),2)

    output_path = os.path.join(args.base_dir, args.output_dir, args.output_name)
    with open(output_path, "w") as outfile: 
        json.dump(results, outfile)


if __name__ == '__main__':
    main()
