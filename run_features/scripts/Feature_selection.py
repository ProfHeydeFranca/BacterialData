#!/usr/bin/python3

## @@ author : Aristeidis Litos

##############################################################  HOW TO   ####################################################################################
############### python Feature_selection.py -t TABLE -f MAX_FEATURES -o OUTDIR -y TARGET --settype MY_BEAUTIFUL_SELECTION


########### TARGET can be either a column of the table 
###########################   OR 
########### a file that contains the same names (first column or row) as the table with a header!! 

######### EXAMPLE OF TARGET TABLE
######    sample	target
######    O2005259	1
######    O2005262	3
######    O2005264	1
######    O2005269	7
######    O2005270	2

'''
NOTE for usage
Lasso if not converge --> GPU (numerical instability)
'''

'''
ABOUT IMPORTANCES


Sequencial forward NOT YET. PLEASE CONTACT ME FIRST

RFE Importance is not A METRIC OF IMPORTANCE. RFE ranks the features based on when (and if) they got eliminated. 
    The metric here is the 1/ranking. Meaning that importance of 1 have all the features that NEVER got eliminated and are the most important.
    0.5 is the metric for the LAST eliminated feature, followed by 0.33 etc. 
    The model that is tested to select the features is a decision tree, that internally uses the Gini criterion of information gain to make decisions.
    If needed, contact me to assess the importances of the tree(s).

L2 importances are the absolute coefficients that SVM(C or R) assigns.

Lasso importances are based on Linear (or Logistic) regression.

Tree-based importances are the importances that RANDOM FOREST assigns to the features. 
    How high a feature is selected in all trees of the forest.
'''


'''
ABOUT METHODS

RFE: Recursive Feature Elimination
    Starts with all features and starts eliminating (and testing a model) one by one up to the max_features argument.


tree-based:
    Starts from 1 feature. Evaluates the next feature based on the information gain GIVEN the previous.
    
'''

##############################################################################################################################################################
###################################################  IMPORTING NECESSARY LIBRARIES  ##########################################################################
##############################################################################################################################################################

import pandas as pd
import os,csv,argparse,time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel, RFECV, SequentialFeatureSelector
from sklearn.linear_model import Lasso,LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC,SVR
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

##############################################################################################################################################################
#########################################################  SPECIFY ARGUMENTS  ################################################################################
##############################################################################################################################################################

## Intoduce arguments
parser = argparse.ArgumentParser()
# Argument for the big table
### Big table should have column AND row names
parser.add_argument("-t","--bigtable", help="Where is the bigtable??",type=str,nargs="?",default='/home/arislit/Downloads/Feature_selection_Bia/df_salt_gene-familiesClassification_0.7.csv')#/home/arislit/Downloads/Feature_selection_Jose/all_table.tsv 
# Argument for the method that you want
# If empty, the script will run all: Lasso, RFE, L1, tree-based, or sequential-forward
parser.add_argument("-m","--method", help="One of Lasso, RFE, L2, tree-based, or sequential-forward??",nargs="?",type=str,default='tree-based')
# How many features should we keep (maximum)
parser.add_argument("-f","--max_features", help="Number of maximum features to select",nargs="?",type=int,default=10000)
# Where are we saving the outputs??
parser.add_argument("-o","--outdir", help="Where should I save the features??",nargs="?",type=str,default='/home/arislit/Downloads/Feature_selection_Bia') #/Feature_selection_Jose
# Which column (or row if columnwise) should the selection be based on? 
### Requires a table with a header (columnnames) or rownames if columnwise
parser.add_argument("-y","--target", help="Which column/feature shoud the selection be based on??\n Can be the path of a csv/tsv",nargs="?",type=str,default='Salinity group') 
# If false, the selection will run on the rows 
parser.add_argument("--columnwise", help="Do you want columns or rows to be selected??",action="store_false",default=True)
## Specify a prefix for every output!!
parser.add_argument("--settype", help="Is there a specific name that you need??",nargs="?",type=str,default='Salt_Bia')
# Do we need a specific separator for your table(s)??
parser.add_argument("--separator", help="Do you want columns or rows to be selected??",default=',')

parser.add_argument("--y_type", help="Do you want columns or rows to be selected??",default=',')



##############################################################################################################################################################
############################################  SETTING THE MODELS,FUNCTIONS AND VARIABLES  ####################################################################
##############################################################################################################################################################

args=parser.parse_args()
# Setting up models
models={
    # If the target variable is continuous, we go here
    'regression':{ 
        'Lasso': Lasso(),
        'RFE': RFECV(estimator=DecisionTreeClassifier(), min_features_to_select=1,cv=10),
        'L2': SVR(C=1.0,kernel='linear'),
        'tree-based': RandomForestRegressor(n_estimators=100)
    
    },
    # For categorical variables we play here
    'classification':{
        'tree-based': RandomForestClassifier(),
        'sequential-forward': SequentialFeatureSelector(DecisionTreeClassifier(), n_features_to_select='best', direction='forward'),
        'Lasso': Lasso(),
        'RFE': RFECV(estimator=DecisionTreeClassifier(), min_features_to_select=1,cv=10),
        'L1': SVC(kernel='linear'),
    }
}

# Setting a unified way to extract the feature importance
feature_importance_methods = {
    'Lasso': lambda model: np.abs(model.coef_),
    'RFE': lambda selector: 1/selector.ranking_,
    'L2': lambda model: np.abs(model.coef_)[0,:],  
    'tree-based': lambda model: model.feature_importances_
    # Add more methods as needed
}

# Define the function that performs the feature selection
def perform_feature_selection(method, X, y,max_features):
    # Time it
    start_time = time.time()
    # Check the target variable type to specify the task
    if np.issubdtype(y.dtype, np.number):
        task_type = 'regression'
        y_encoded=y
    else:
        task_type = 'classification'
        # Convert categorical labels to numerical values using LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

    # Define the model to use
    model=models[task_type][method]
    # Fit the model
    model.fit(X,y_encoded)
    # Extract the importance
    importance=feature_importance_methods[method](model)
    # Select the top k indexes
    print (np.argsort(importance)[::-1])
    top_k_indices = np.argsort(importance)[::-1][:max_features] #if method!='L2' else  np.argsort(importance)[::-1][0,:][:max_features] 
    # Select the top k features by the top k indexes
    top_k_features = list(set([X.columns[i] for i in top_k_indices]))
    # Stop timer
    end_time = time.time()
    # Calculate the time used
    elapsed_time = end_time - start_time

    return  top_k_features, top_k_indices, importance,elapsed_time #if method!='L2' else  np.argsort(importance)[::-1][0,:], elapsed_time

# Set all methods
methods = ['Lasso', 'RFE', 'L2', 'tree-based']#, 'sequential-forward']
# Define label encoder
label_encoder = LabelEncoder()
##############################################################################################################################################################
##############################################  READING ARGUMENTS, TABLES, PERFORMING  #######################################################################
##############################################################################################################################################################


# Proprerly read the output directory
outdir=args.outdir if args.outdir[-1]=='/' else args.outdir+'/'
# Read the big table
with open(args.bigtable, 'r') as csvfile:
    sample = csvfile.read(512)  
    # Use the Sniffer to Detect the separator
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(sample)

    # Retrieve the Detected Separator
    separator = dialect.delimiter


bigtable=pd.read_csv(args.bigtable,sep=separator,header=0,index_col=0) if args.columnwise else pd.read_csv(args.bigtable,sep=separator,header=0,index_col=0).T
'''







'''
#bigtable=bigtable[list(bigtable.columns)[:500]+[args.target]]

'''



'''



if not os.path.isfile(args.target):
    # Find the target variable
    y = bigtable[args.target]
    # Drop the target variable
    X = bigtable.drop(columns=[args.target])
else:
    try:
        with open(args.target, 'r') as csvfile:
            sample = csvfile.read(512)  
            # Use the Sniffer to Detect the separator
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)

            # Retrieve the Detected Separator
            separator2 = dialect.delimiter
                
        y=pd.read_csv(args.target,sep=separator2,header=0,index_col=0) if args.columnwise else pd.read_csv(args.target,sep=separator2,header=0,index_col=0).T
    except Exception:
        raise ValueError(f'Invalid option for y {args.target}')
    
    X=bigtable
    # Rearrange y to the correct order
    y=y.loc[X.index.tolist()]
# Transorm y properly
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Set the prefix
settype='_'+args.settype if args.settype else args.settype

# If you selected a specific method
if args.method:
    # Perform the feature selection for that method
    selected_features, selected_indeces, importance, _ = perform_feature_selection(args.method, X, y_encoded,args.max_features)
    # Initialize a dataframe to save the k important features
    importances=pd.DataFrame([])
    # Import the features in the dataframe
    importances['Feature']=selected_features
    # Import the score as well
    importances['Importance_score']=np.sort(importance)[-args.max_features:]
    # Save the tsv of the important features
    importances.to_csv(f'{outdir}Importance_scores_{args.method}{settype}.tsv',sep='\t',header=True,index=False)
    # Define the final table from the original
    bigtable[selected_features].to_csv(f'{outdir}Selected_table.tsv',header=True,index=True)
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x= importances['Feature'],y=importances['Importance_score'])
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{outdir}Importance_{args.method}{settype}.png')
    #plt.show()

# If you didn't select a method, we're gonna run them all!!
else:
    # Initialize a dictionary to save results for all methods
    sel={method:{} for method in ['Lasso', 'L2', 'tree-based', 'RFE']}
    #
    sel=pd.DataFrame(columns=['Lasso', 'L2', 'tree-based', 'RFE'],index=X.columns)
    # Iterate through the methods
    for method in ['Lasso', 'L2', 'tree-based', 'RFE']:
        # Feature selection for a method
        selected_features, selected_indeces, importance, _ = perform_feature_selection(method, X, y_encoded,args.max_features)
        # Save the importance values to the dataframe
        for i in range(len(selected_features)):
            sel.loc[selected_features[i],method]=importance[i]

        # Initialize a dataframe per method
        importances=pd.DataFrame([])
        # Import the features in the dataframe
        importances['Feature']=selected_features
        # Import the score as well
        importances['Importance_score']=np.sort(importance)[-args.max_features:]
        # Save the tsv of the important features
        importances.to_csv(f'{outdir}Importance_scores_{args.method}{settype}.tsv',sep='\t',header=True,index=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x= importances['Feature'],y=importances['Importance_score'])
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title(f'Feature Importance {method}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{outdir}Imprortance_{method}{settype}.png')
        # Print the progress of the algorithm
        print (f'Done with {method}')
    # Save the tsv of ALL important features
    sel.to_csv(f'{outdir}Important_features_all{settype}.tsv',sep='\t',header=False,index=False)

print (f'Thank you very much for selecting features with us! \n Your results are saved in {outdir}')