import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import r_regression
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder,RobustScaler,PowerTransformer,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy import stats

class rnCV():
    def __init(self,r=10,n=5,k=3):
        self.R=r
        self.N=n
        self.K=k


def clean_data(data:pd.DataFrame):
    df=data

    general_info(df)
    nan_columns=df.columns[df.isna().any()].tolist()
    total_nan=(df.isna().sum()).sum()

    df_numeric = df.select_dtypes(include=[float, int])
    df_numeric=df_numeric.drop(columns=['id'])

    print(f'Our data consists of {df.shape[1]} columns and {df.shape[0]} entries')
    print(f'We have {df_numeric.shape[1]} numeric columns. These are {list(df_numeric.columns)}')
    print(f'{len(nan_columns)} columns have missing values. These columns are : {nan_columns}')
    print(f'In total we have {total_nan} missing values')

    df=impute(data_df=df)

    nan_columns=df.columns[df.isna().any()].tolist()
    print(f'We now have {len(nan_columns)} columns with missing values. These columns are : {nan_columns}')
    
    df=encode(data_df=df,target='diagnosis')

    df=remove_duplicates(df)
    check_for_outliers(df)
    print("We WON'T remove the outliers.")

    return df


def general_info(data_df: pd.DataFrame):
    print(f'Shape of dataset: {data_df.shape} ({data_df.shape[0]} entries and {data_df.shape[1]} columns)')
    print(f'Data type of the {data_df.shape[1]} columns\n {data_df.dtypes}')

def impute(data_df: pd.DataFrame):
    df=data_df

    df_numeric = df.select_dtypes(include=[float, int])
    
    if 'id' in df_numeric.columns:
        df_numeric=df_numeric.drop(columns=['id'])

    imp = IterativeImputer(random_state=42)
    df[df_numeric.columns] = imp.fit_transform(df_numeric)

    return df

def encode(data_df: pd.DataFrame,target='diagnosis'):
    """
    Method used to encode the entries of the column 'diagnosis'
    Malignant --> 1
    Benign --> 0
    """
    df=data_df
    
    df[target]=LabelEncoder().fit_transform(df[target]) # M->1, B->0

    return df
    

def remove_duplicates(data_df: pd.DataFrame):
    """
    We use this function to find and remove any potential duplicates
    """
    
    df=data_df
   
    shape_before=df.shape
    df.drop_duplicates()
    shape_after=df.shape

    if (shape_before[0] != shape_after[0]):
        print("Before removal of duplicates",shape_before)
        print("After removal of duplicates",shape_after)
    else:
        print("No duplicates in the set")
    
    return df

def check_for_outliers(data_df: pd.DataFrame):

    df=data_df
   
    shape_before=df.shape
    
    no_outliers=df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

    shape_after=no_outliers.shape

    if (shape_before[0] != shape_after[0]):
        removed=shape_before[0]-shape_after[0]
        print("Before removal of outliers",shape_before)
        class_imbalance(df)
        print("After removal of outliers",shape_after)
        class_imbalance(no_outliers)
        print(f"We could remove {removed} entries ({(removed/shape_before[0])*100:.2f}% of total entries)")

    else:
        print("No outliers in the set")
    
def class_imbalance(data_df: pd.DataFrame,field='diagnosis'):
    df=data_df
    order=[0,1]

    entries=data_df[field].value_counts().reindex(order)
    print(f'Absolute frequencies of field "{field}"')
    print(entries)

    fractions=data_df[field].value_counts(normalize=True)
    print(f'Percentage of each class of field "{field}"')
    print(fractions)


def get_Y(data_df: pd.DataFrame,target='diagnosis'):
    if target not in data_df.columns:
        raise ValueError("Please give a valid target!")
    
    return data_df[target]

def keep_features(data_df: pd.DataFrame,target='diagnosis',to_drop=None):
    tdrp=[]
    
    if to_drop is not None:
        tdrp = [col for col in to_drop if col in data_df.columns]
    
    tdrp.append(target)
    Y=get_Y(data_df=data_df,target=target)
    x=data_df.drop(tdrp,axis=1)

    return x,Y

def corr_between_target(data_df: pd.DataFrame,target='diagnosis',thres=0.1):
    x,Y=keep_features(data_df=data_df,target=target,to_drop=['id'])
    corr=pd.Series(r_regression(x,Y),index=x.columns)
    selected=corr[corr.abs() >= thres].index.tolist()

    print("We could keep only the most correlated features:")
    print(selected)

    print(f'If we do, we will go from {x.shape[1]} features to {len(selected)} features,that will be the most correlated')

    print("Returning the features that could be kept")
    viz_corr_between_target(corr=corr,target='diagnosis')


    return selected

def corr_between_features(data_df: pd.DataFrame,target='diagnosis',to_drop=['diagnosis','id'],thres=0.8):
    df=data_df

    feats=df.drop(columns=to_drop).columns.to_list()

    corr_matrix=df[feats].corr(method='pearson')

    viz_corr_between_features(corr_matrix=corr_matrix)
    
    corr_pairs= corr_matrix.abs().where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().sort_values(ascending=False)

    high_pairs=corr_pairs[corr_pairs>=thres]
    print('Pairs of high correlation')
    print(high_pairs)

    high_to_drop=set()

    for high_pair in high_pairs.index:
        high_to_drop.add(high_pair[1])
    
    print("We could remove these features:")
    print(high_to_drop)

    high_selected=df.drop(columns=high_to_drop)
    print(f"If we do, we will go from {len(feats)} features to {len(high_selected.columns)} features")
    print('Returning the features that could be ignored')

    return high_to_drop




def viz_corr_between_features(corr_matrix):
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap=sns.cubehelix_palette(as_cmap=True), center=0,linewidth=.5)
    plt.title('Heatmap of Correlation between Features')
    plt.show()

def viz_corr_between_target(corr:pd.Series,target='diagnosis'):
    plt.figure(figsize=(10,6))
    corr.sort_values().plot(kind='barh',color='salmon')
    plt.title(f"Feature Correlations with {target}")
    plt.xlabel("Pearson's Correlation Coefficient")
    plt.ylabel("Feature")
    plt.axvline(0, color='black', linestyle='--')
    plt.show()

def boxpolt_distro(data_df: pd.DataFrame,to_drop=['diagnosis','id']):
    df=data_df

    feats=df.drop(columns=to_drop).columns.to_list()

    plt.figure(figsize=(12,30))

    for i,feat in enumerate(feats,1):
        plt.subplot(10, 3, i)
        sns.boxplot(x=df[feat], color='salmon',flierprops={"marker":"x"})
        plt.tight_layout()
    plt.suptitle("Feature Distributions", fontsize=14, y=1.02)
    plt.show()

def perform_pca(data_df: pd.DataFrame):
    x,Y=keep_features(data_df=data_df,target='diagnosis',to_drop=['id'])

    data_rescaled=scale_data(x)

    fraction=0.95

    pca = PCA(n_components = fraction)
    pca.fit(data_rescaled)
    reduced = pca.transform(data_rescaled)

    print(f"{fraction*100}% of the variance can be explained by {pca.n_components_} components")
    print("The explained variance ratio is: ",(pca.explained_variance_ratio_))

    viz_pca(reduced,Y)

    pca_df=pd.DataFrame(data=reduced)
    pca_df['diagnosis']=Y.values
    
    return pca_df

def scale_data(data):
    pipeline = Pipeline([
    ('scaler', RobustScaler()),  
    ('transformer', PowerTransformer(method='yeo-johnson'))
    ])
    data_rescaled = pipeline.fit_transform(data)

    return data_rescaled

def viz_pca(x,y):
    plt.figure(figsize=(10, 8))
    scat = plt.scatter(x[:, 0], x[:, 1], c=y.values, cmap='flare', alpha=0.7)

    plt.legend(*scat.legend_elements(), title="Labels")
    plt.title("Principal Component Analysis")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()