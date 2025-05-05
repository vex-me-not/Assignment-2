import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import optuna
import lightgbm as lgb
import warnings

from pathlib import Path
from optuna.pruners import MedianPruner
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import r_regression
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder,RobustScaler,PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score, balanced_accuracy_score,f1_score, precision_score, recall_score, confusion_matrix, average_precision_score
from scipy import stats


warnings.filterwarnings('ignore')

class IO:
    """
    This class implements the saving and the loading of the models. It needs a models directory in the constructor to work
    So, each instance of IO class coresponds to one specific directory.
    If you want to load or to save from another directory, you need to create a new instance
    """
    def __init__(self,models_dir):
        if models_dir is None:
            raise ValueError("Directory of models must be given!")

        self.models_dir=Path(models_dir)
        os.makedirs(self.models_dir,exist_ok=True)

    def __gen_filename(self,name,suffix):
        parts=[name]
        if suffix:
            parts.append(suffix)

        return self.models_dir/f"{'_'.join(parts)}.pkl"
    
    # method that saves a model adding a suffix suf to its name
    def save(self,model,name,suf=''):
        joblib.dump(model,self.__gen_filename(name,suffix=suf))

    # method
    def load(self,name,suf=''):
        return joblib.load(self.__gen_filename(name,suffix=suf))

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.best = None
        self.no_impr_count = 0

    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        if (self.best is None) or (study.best_value > self.best):
            self.best = study.best_value
            self.no_impr_count = 0
        else:
            self.no_impr_count += 1

        if (self.no_impr_count >= self.patience):
            print(f"EARLY STOPPING: No improvement after {self.patience} trials!")
            study.stop()
            print(f"--> Best trial is {study.best_trial.number} with value: {study.best_trial.value} and parameters: {study.best_trial.params}\n")



class rnCV():
    def __init__(self,data_df,estimators,params,r=10,n=5,k=3,random_state=42):
        self.R=r
        self.N=n
        self.K=k
        self.x, self.y=keep_features(data_df=data_df,target='diagnosis',to_drop=None)
        self.y= encode(self.y)
        self.estimators=estimators
        self.params=params
        self.random_state=random_state
        self.results={estim:[] for estim in estimators.keys()}

    def _compute_metrics(self,y_true,y_pred,y_prob):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        npv = tn / (tn + fn)

        metrics={
            'MCC': matthews_corrcoef(y_true, y_pred),
            'ROC_AUC': roc_auc_score(y_true, y_prob),
            'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Specificity': specificity,
            'NPV': npv,
            'PR_AUC': average_precision_score(y_true, y_prob)
        }

        return metrics
    
    def results_summary(self):
        summary = {}
        
        for model_name, all_metrics in self.results.items():
            df = pd.DataFrame(all_metrics)
            summary[model_name] = df.median().to_dict()
    
        summary_t=pd.DataFrame(summary).T
        
        return summary_t

    def _tune_model(self,x_train,y_train,model_name):
        # per optuna's documentation
        def objective(trial):
            estimator=self.estimators[model_name]
            params=self.params[model_name](trial)
            model=estimator(**params)

            strat_kfold=StratifiedKFold(n_splits=self.K, shuffle=True, random_state=self.random_state)
            scores = []

            for fold_idx, (k_train_idx, k_val_idx) in enumerate(strat_kfold.split(x_train, y_train)):
                x_k_train, x_val = x_train.iloc[k_train_idx], x_train.iloc[k_val_idx]
                y_k_train, y_val = y_train.iloc[k_train_idx], y_train.iloc[k_val_idx]
                
                x_k_train=impute(x_k_train)
                x_k_train=scale_data(x_k_train)
                
                x_val=impute(x_val)
                x_val=scale_data(x_val)

                model.fit(x_k_train, y_k_train)
                preds = model.predict(x_val)
                score=balanced_accuracy_score(y_val, preds)
                scores.append(score)

                # report intermediate result for pruning
                trial.report(score, step=fold_idx)
                
                # ask if the trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)        
        

        study = optuna.create_study(direction='maximize',study_name=model_name)
        study.optimize(objective, n_trials=60,timeout=180.0,callbacks=[EarlyStopping(patience=10)])
        
        return study.best_params        

    def run_rnCV(self):
        np.random.seed(self.random_state)
        
        for r in range(self.R):
            print(f"------ Repetition {r+1}/{self.R} ------\n")

            state=self.random_state + r
            strat_kfold=StratifiedKFold(n_splits=self.N, shuffle=True, random_state=state)

            for model_name in self.estimators.keys():
                for n_train_idx, n_test_idx in strat_kfold.split(self.x, self.y):
                    x_n_train, x_n_test = self.x.iloc[n_train_idx], self.x.iloc[n_test_idx]
                    y_n_train, y_n_test = self.y.iloc[n_train_idx], self.y.iloc[n_test_idx]


                    best_params = self._tune_model(x_train=x_n_train,y_train= y_n_train,model_name=model_name)

                    model = self.estimators[model_name](**best_params)

                    x_n_train=impute(x_n_train)
                    x_n_train=scale_data(x_n_train)
                    
                    x_n_test=impute(x_n_test)
                    x_n_test=scale_data(x_n_test)
                    
                    model.fit(x_n_train, y_n_train)

                    preds = model.predict(x_n_test)
                    pred_probs = model.predict_proba(x_n_test)[:, 1]

                    metrics = self._compute_metrics(y_true=y_n_test,y_pred=preds,y_prob=pred_probs)
                    self.results[model_name].append(metrics)

        return self.results

    def tune_winner(self,winner):

        # per optuna's documentation
        def objective(trial):
            estimator=self.estimators[winner]
            params=self.params[winner](trial)
            model=estimator(**params)

            strat_kfold=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = []

            x_tune=self.x
            y_tune=self.y
            
            for fold_idx, (tune_train_idx, tune_val_idx) in enumerate(strat_kfold.split(x_tune, y_tune)):
                x_tune_train, x_tune_val = x_tune.iloc[tune_train_idx], x_tune.iloc[tune_val_idx]
                y_tune_train, y_tune_val = y_tune.iloc[tune_train_idx], y_tune.iloc[tune_val_idx]
                    
                x_tune_train=impute(x_tune_train)
                x_tune_train=scale_data(x_tune_train)
                    
                x_tune_val=impute(x_tune_val)
                x_tune_val=scale_data(x_tune_val)

                model.fit(x_tune_train, y_tune_train)
                preds = model.predict(x_tune_val)
                score=balanced_accuracy_score(y_tune_val, preds)
                scores.append(score)

                # report intermediate result for pruning
                trial.report(score, step=fold_idx)
                    
                # ask if the trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)        
            

        study = optuna.create_study(direction='maximize',study_name="Winner:"+winner)
        study.optimize(objective, n_trials=60,timeout=180.0,callbacks=[EarlyStopping(patience=10)])
            
        return study.best_params



def perform_rnCV(path):
    df=pd.read_csv(path)
    # Define estimators
    estimators = {
        'LogisticRegression': LogisticRegression,
        'GaussianNB': GaussianNB,
        'LDA': LinearDiscriminantAnalysis,
        'SVM': SVC,
        'RandomForest': RandomForestClassifier,
        'LightGBM': lgb.LGBMClassifier
    }

    # Define hyperparameter spaces
    param_spaces = {
        'LogisticRegression': lambda trial: {
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
            'solver': trial.suggest_categorical('solver', ['saga']),
            'C': trial.suggest_float('C', 1e-3, 1e0, log=True),
            'l1_ratio': trial.suggest_uniform('l1_ratio', 0, 1)
        },
        'GaussianNB': lambda trial: {'var_smoothing': trial.suggest_float('var_smoothing', 1e-2, 1e-1, log=True)},
        'LDA': lambda trial: {'solver':trial.suggest_categorical('solver', ['svd', 'lsqr', 'eigen']),
                              'tol':trial.suggest_float('tol', 5*1e-2, 1e-1, log=True)},
        'SVM': lambda trial: {
            'C': trial.suggest_float('C', 5*1e-2, 1e2, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
            'probability': trial.suggest_categorical('probability', [True])
        },
        'RandomForest': lambda trial: {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
        },
        'LightGBM': lambda trial: {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'verbosity': trial.suggest_categorical('verbosity', [-1])
        }
    }

    # Initialize and run rnCV
    rncv = rnCV(data_df=df, estimators=estimators,params=param_spaces, r=10, n=5, k=3, random_state=42)
    results = rncv.run_rnCV()

    # Summarize and save the results
    summary = rncv.results_summary()
    summary.to_csv('../data/rncv_summary_results.csv')

    print("Summary:\n", summary)

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

    entries=df[field].value_counts().reindex(order)
    print(f'Absolute frequencies of field "{field}"')
    print(entries)

    fractions=df[field].value_counts(normalize=True)
    print(f'Percentage of each class of field "{field}"')
    print(fractions)


def get_Y(data_df: pd.DataFrame,target='diagnosis'):
    if target not in data_df.columns:
        raise ValueError("Please give a valid target!")
    
    return pd.DataFrame(data_df[target])

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

def count_apps(item,item_list:list):
    count=0
    for it in item_list:
        if (it==item):
            count += 1
    return count

def count_wins(summary:pd.DataFrame):
    total_models=summary.shape[0]

    dict_app={}
    for model_num in range(total_models):
        dict_app[model_num]=count_apps(model_num,summary.idxmax())
    
    return dict_app

def winner_dict(summary:pd.DataFrame):
    
    model_names=summary['Model']
    sum_no_name=summary.drop(columns='Model')

    win_dict=count_wins(summary=sum_no_name)

    for i in range(len(model_names)):
        win_dict[model_names[i]]=win_dict.pop(i)

    return win_dict

def get_winner(summary:pd.DataFrame):
    win_dict=winner_dict(summary=summary)

    sorted_win_dict=dict(sorted(win_dict.items(), key=lambda item: item[1],reverse=True))
    winner=list(sorted_win_dict.keys())[0]

    return (winner,sorted_win_dict[winner])

def replace_column(df:pd.DataFrame,to_be_replaced,to_be_added):
    data_df=df
    
    if to_be_replaced in data_df.columns and (to_be_added not in data_df.columns):
        data_df.rename(columns={to_be_replaced:to_be_added},inplace=True)
    
    return data_df  


def winner_tuning(df:pd.DataFrame,winner):
    # Define estimators
    estimators = {
        'LogisticRegression': LogisticRegression,
        'GaussianNB': GaussianNB,
        'LDA': LinearDiscriminantAnalysis,
        'SVM': SVC,
        'RandomForest': RandomForestClassifier,
        'LightGBM': lgb.LGBMClassifier
    }

    # Define hyperparameter spaces
    param_spaces = {
        'LogisticRegression': lambda trial: {
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
            'solver': trial.suggest_categorical('solver', ['saga']),
            'C': trial.suggest_float('C', 1e-3, 1e0, log=True),
            'l1_ratio': trial.suggest_uniform('l1_ratio', 0, 1)
        },
        'GaussianNB': lambda trial: {'var_smoothing': trial.suggest_float('var_smoothing', 1e-2, 1e-1, log=True)},
        'LDA': lambda trial: {'solver':trial.suggest_categorical('solver', ['svd', 'lsqr', 'eigen']),
                              'tol':trial.suggest_float('tol', 5*1e-2, 1e-1, log=True)},
        'SVM': lambda trial: {
            'C': trial.suggest_float('C', 5*1e-2, 1e2, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
            'probability': trial.suggest_categorical('probability', [True])
        },
        'RandomForest': lambda trial: {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
        },
        'LightGBM': lambda trial: {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'verbosity': trial.suggest_categorical('verbosity', [-1])
        }
    }

    rncv = rnCV(data_df=df, estimators=estimators,params=param_spaces, r=10, n=5, k=3, random_state=42)
    winner_estim=estimators[winner]
    winner_params=rncv.tune_winner(winner=winner)

    print(f'For model {winner} the best parameters are {winner_params}')

    return winner_estim(**winner_params)


def metric_ci(y_true, y_pred, metric, is_proba=False, proba=None, n_samples=5000, seed=42):
    
    rng = np.random.RandomState(seed)
    stats = []
    
    for sample in range(n_samples):
        
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        
        y_sample = y_true[idx] if isinstance(y_true, np.ndarray) else y_true.iloc[idx]
        pred_sample = proba[idx] if is_proba else y_pred[idx]
        
        stats.append(metric(y_sample, pred_sample))
    
    return np.percentile(stats, [2.5, 97.5])


def bootstrap_model_intervals(df_dev:pd.DataFrame,df_val:pd.DataFrame, model):
    metrics = {
        'Balanced Accuracy': (balanced_accuracy_score, False),
        'F1 Score': (f1_score, False),
        'Precision': (precision_score, False),
        'Recall': (recall_score, False),
        'MCC': (matthews_corrcoef, False),
        'ROC AUC': (roc_auc_score, True),
        'PR AUC': (average_precision_score, True)
    }


    x_dev, y_dev=keep_features(data_df=df_dev,target='diagnosis',to_drop=None)
    y_dev = encode(y_dev)


    x_dev=impute(x_dev)
    x_dev=scale_data(x_dev)
    
    x_val, y_val=keep_features(data_df=df_val,target='diagnosis',to_drop=None)
    y_val = encode(y_val)

    x_val=impute(x_val)
    x_val=scale_data(x_val)

    model.fit(x_dev,y_dev)

    # predict on evaluation set
    y_pred = model.predict(x_val)
    y_proba = model.predict_proba(x_val)[:, 1]


    print("Bootstrapped 95% CIs (Model trained on dev, tested on val):")
    for name, (fn, is_proba) in metrics.items():
        ci = metric_ci(
            y_val,
            y_pred if not is_proba else y_proba,
            fn,
            is_proba=is_proba,
            proba=y_proba if is_proba else None
        )
        print(f"{name:15s}: [{ci[0]:.4f}, {ci[1]:.4f}]")

    # specificity and NPV as point estimates (not bootstrapped here)
    tn, fp, fn_, tp = confusion_matrix(y_val, y_pred).ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn_) if (tn + fn_) > 0 else 0
    
    print(f"Specificity         : {specificity:.4f}")
    print(f"NPV                 : {npv:.4f}")


def metric_ci_plot(y_true, y_pred, proba, n_samples=1000, seed=42):
    rng = np.random.RandomState(seed)
    metrics = {
        'Balanced Accuracy': [],
        'F1 Score': [],
        'Precision': [],
        'Recall': [],
        'MCC': [],
        'ROC AUC': [],
        'PR AUC': []
    }

    for _ in range(n_samples):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        y_bs = y_true[idx] if isinstance(y_true, np.ndarray) else y_true.iloc[idx]
        y_pred_bs = y_pred[idx]
        y_proba_bs = proba[idx]

        metrics['Balanced Accuracy'].append(balanced_accuracy_score(y_bs, y_pred_bs))
        metrics['F1 Score'].append(f1_score(y_bs, y_pred_bs))
        metrics['Precision'].append(precision_score(y_bs, y_pred_bs))
        metrics['Recall'].append(recall_score(y_bs, y_pred_bs))
        metrics['MCC'].append(matthews_corrcoef(y_bs, y_pred_bs))
        metrics['ROC AUC'].append(roc_auc_score(y_bs, y_proba_bs))
        metrics['PR AUC'].append(average_precision_score(y_bs, y_proba_bs))

    return metrics

def bootstrap_model_plot(df_dev:pd.DataFrame,df_val:pd.DataFrame, model):

    x_dev, y_dev=keep_features(data_df=df_dev,target='diagnosis',to_drop=None)
    y_dev = encode(y_dev)


    x_dev=impute(x_dev)
    x_dev=scale_data(x_dev)
    
    x_val, y_val=keep_features(data_df=df_val,target='diagnosis',to_drop=None)
    y_val = encode(y_val)

    x_val=impute(x_val)
    x_val=scale_data(x_val)

    model.fit(x_dev,y_dev)

    # predict on evaluation set
    y_pred = model.predict(x_val)
    y_proba = model.predict_proba(x_val)[:, 1]

    # Run bootstrapping
    bootstrap_scores = metric_ci_plot(y_val, y_pred, y_proba, n_samples=5000)

    # Visualize with violin plot
    df_bootstrap = pd.DataFrame({
        'Metric': sum([[metric]*len(scores) for metric, scores in bootstrap_scores.items()], []),
        'Score': sum(bootstrap_scores.values(), [])
    })

    # Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_bootstrap, x='Metric', y='Score', inner='quartile', palette='pastel')
    plt.title("Bootstrapped 95% CI Distributions (val set)")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def bootstrap_model(df_dev:pd.DataFrame,df_val:pd.DataFrame, model):
    bootstrap_model_intervals(df_dev=df_dev,df_val=df_val,model=model)
    bootstrap_model_plot(df_dev=df_dev,df_val=df_val,model=model)

def save_winner(winner,winner_name):
    models_dir="../models"
    model_io=IO(models_dir)

    print(f"Saving winner model with name {winner_name}")

    model_io.save(model=winner,name=winner_name)