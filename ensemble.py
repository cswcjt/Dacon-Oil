# ---------------------------------------------------------------------------- #
#                 Make Model Pipeline for the multiple tests                   #
# ---------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import warnings
warnings.filterwarnings(action='ignore')

# classification models
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from lightgbm.sklearn import LGBMClassifier
from xgboost.sklearn import XGBClassifier

# classification metrics
from sklearn.metrics import accuracy_score, f1_score, auc

# regression models
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBRegressor

# regression metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error, r2_score

# KFold(CV), partial : optuna를 사용하기 위함
from sklearn.model_selection import StratifiedKFold, KFold

# optimize : hyper-parameter tuning
from sklearn.model_selection import train_test_split, GridSearchCV
import optuna
from optuna import Trial
from optuna.samplers import TPESampler

class Ensemble:
    """
    Ensemble & Optimize 3 models(RandomForest, XGBoost, LightGBM)
    - (model_type : Classifier / Regressor) Ensemble with (ensemble : voting / stacking)
    """
    def __init__(self, metric: str, learner: str='auto', ensemble: str='voting'):
        """
        metric : sklearn.metrics 내장함수 활용
        learner : 'auto' - rf, xgb, lgbm ensemble / 'rf' - RandomForest / 'xgb' - XGBoost / 'lgbm' - LightGBM
        ensemble : 'voting', 'stacking'
        """
        self.final_ensemble = None        # Final Ensemble model
        self.models = {
            'rf': None,
            'xgb': None,
            'lgbm': None
        }
        
        # self.metric_dict[self.type_][self.metric_]
        self.metric_dict = {
            'classification': {
                'accuracy_score': accuracy_score,
                'f1_score': f1_score,
                'auc': auc,
            },
            
            'regression': {
                'mae': mean_absolute_error,
                'mse': mean_squared_error,
                'rmse': mean_squared_error,
                'msle': mean_squared_log_error,
                'rmsle': mean_squared_log_error,
                'mape': mean_absolute_percentage_error,
                'r2_score': r2_score,
            }
        }

        # self.metric_direction_dict[self.type_][self.metric_]
        self.metric_direction_dict = {
            'classification': {
                'accuracy_score': 'maximize',
                'f1_score': 'maximize',
                'auc': 'maximize'
            },

            'regression': {
                'mae': 'minimize',
                'mse': 'minimize',
                'rmse': 'minimize',
                'msle': 'minimize',
                'rmsle': 'minimize',
                'mape': 'minimize',
                'r2_score': 'maximize'
            }
        }

        # Initializing hyper-parameter for each model
        # self.param[self.learner_]
        self.param = {
            'rf' : {'n_jobs': -1,
                    'random_state': 42},
            
            'xgb' : {'learning_rate': 0.001,
                     'nthread' : -1,
                     'n_jobs': -1,
                     'tree_method': 'gpu_hist',
                     'predictor': 'gpu_predictor',
                     'random_state': 42},
            
            'lgbm' : {'learning_rate': 0.001,
                      'n_jobs': -1,
                      'random_state': 42}
        }

        # self.learners[self.type_][self.learner_]
        self.learners = {
            'classification' : {
                'rf': RandomForestClassifier,
                'xgb': XGBClassifier,
                'lgbm': LGBMClassifier
            },
            
            'regression' : {
                'rf': RandomForestRegressor,
                'xgb': XGBRegressor,
                'lgbm': LGBMRegressor
            }
        }

        # self.voters[self.type_][self.emsemble_]
        self.voters = {
            'classification' : {
                'voting' : VotingClassifier,
                'stacking' : StackingClassifier
            },
            
            'regression' : {
                'voting' : VotingRegressor,
                'stacking' : StackingRegressor
            }
        }

        # 'classification' , 'regression'
        self.type_ = ''
        self.learner_ = ['rf', 'xgb', 'lgbm'] if learner == 'auto' else [learner]
        self.ensemble_ = ensemble if ensemble in ['voting', 'stacking'] else 'voting'
        self.metric_ = metric

    def make_weights(self, N: int) -> list:
        # x+y+z = 5인 음이 아닌 정수 (x, y, z) 순서쌍 만들기
        weights = []
        for i in range(N+1):
            for j in range(N+1-i):
                k = N-i-j
                temp = [i/N, j/N, k/N]
                weights.append(temp)
        return weights
        
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            n_trials: int=20, cv: int=5, N: int=5) -> None:

        self.X_cols = X_train.columns

        for learner in self.learner_:
            # RF, XGB, LGBM 순서대로 hyper-parameter tuning
            param = self.optimizer(X_train, y_train, learner, n_trials, cv)
            # Hyper-parameter fix + tuning
            self.param[learner].update(param)
            # Set up final models
            self.models[learner] = self.learners[self.type_][learner](**self.param[learner])
            self.models[learner].fit(X_train, y_train)

        if len(self.learner_) < 2:
            self.final_ensemble = self.models[self.learner_[0]]

        else:
            estimators = [(learner, self.models[learner]) for learner in self.learner_]
            weights = self.make_weights(N)

            # 'weights': weights,
            ensemble_param = {
                'estimators': estimators,
                'n_jobs': -1
            }

            if self.ensemble_ == 'voting':
                ensemble_param.update({'voting': 'soft'})
            
            self.final_ensemble = self.voters[self.type_][self.ensemble_](**ensemble_param)

            grid_params = {'weights': weights}
            grid_Search = GridSearchCV(param_grid = grid_params, estimator=self.final_ensemble, scoring=self.metric_dict[self.type_][self.metric_])
            grid_Search.fit(X_train, y_train)
            self.final_ensemble = grid_Search.best_estimator_

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.final_ensemble.predict(X_test)

    def score(self, y_true: np.ndarray, y_pred: np.ndarray, cv: int=1) -> float:
        if cv <= 1:
            if self.metric_ in ['rmse', 'rmsle']:
                return self.metric_dict[self.type_][self.metric_[1:]](y_true, y_pred, squared=False)
            else:
                return self.metric_dict[self.type_][self.metric_](y_true, y_pred)

        else:
            folds = KFold(n_splits=cv, shuffle=True, random_state=42)
            scores = []

            for val_idx, _ in folds.split(y_true, y_pred):
                y_true_ = y_true[val_idx]
                y_pred_ = y_pred[val_idx]
                score = self.score(y_true_, y_pred_)
                scores.append(score)

            return np.mean(scores)

    def K_fold(self, model, X, y, cv) -> list:
        scores = []
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        try:
            for train_idx, val_idx in folds.split(X, y):
                continue
        except:
            folds = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        if cv == 1:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            score = self.score(y_val, model.predict(X_val))
            scores.append(score)

        else:
            for train_idx, val_idx in folds.split(X, y):
                X_train = X.iloc[train_idx, :]
                y_train = y.iloc[train_idx]
                
                X_val = X.iloc[val_idx, :]
                y_val = y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                score = self.score(y_val, model.predict(X_val))
                scores.append(score)

        return scores

    def objective(self, trial: Trial, X, y, learner: str, cv: int) -> float:
        temp = copy.deepcopy(self.param[learner])
        
        if learner == 'rf': # RandomForest
            param = {
                "n_estimators" : trial.suggest_int('n_estimators', 50, 1000),
                'max_depth':trial.suggest_int('max_depth', 8, 16),
                'min_samples_split': trial.suggest_int('min_samples_split', 3, 50),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            }

        elif learner == 'xgb': # XGB
            param = {
                "n_estimators" : trial.suggest_int('n_estimators', 500, 4000),
                'max_depth':trial.suggest_int('max_depth', 8, 16),
                'min_child_weight':trial.suggest_int('min_child_weight', 1, 300),
                'gamma':trial.suggest_int('gamma', 1, 3),
                'learning_rate': 0.05,
                'colsample_bytree':trial.suggest_discrete_uniform('colsample_bytree',0.5, 1, 0.1),
                'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
                'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
                'subsample': trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0] ),
            }

        elif learner == 'lgbm': # LGBM
            param = {
                'num_leaves': trial.suggest_int('num_leaves', 2, 1024, step=1, log=True), 
                'max_depth': trial.suggest_int('max_depth', 1, 10, step=1, log=False), 
                'learning_rate': 0.05,
                'n_estimators': trial.suggest_int('n_estimators', 8, 1024, step=1, log=True), 
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50, step=1, log=False), 
                'subsample': trial.suggest_uniform('subsample', 0.7, 1.0), 
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_uniform('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_uniform('reg_lambda', 0.0, 10.0),
            }

        else:
            raise Exception("Not exist those model. Please choose the number in [0, 1, 2]\nTry again.")
        
        # Set up param
        temp.update(param)
        param = temp

        # Set up the model by flag
        model = self.learners[self.type_][learner](**param)
        
        # K-fold cross validation
        scores = self.K_fold(model, X, y, cv)

        return np.mean(scores)
    
    def optimizer(self, X: pd.DataFrame, y: np.ndarray,
                  learner: str, n_trials: int, cv: int) -> dict:
        
        direction = self.metric_direction_dict[self.type_][self.metric_]
        study = optuna.create_study(direction=direction, sampler=TPESampler())
        study.optimize(lambda trial : self.objective(trial, X, y, learner, cv), n_trials=n_trials)
        print('Best trial: score {},\nparams: {}'.format(study.best_trial.value, study.best_trial.params))
        return study.best_trial.params

class BinaryCalssifier(Ensemble):
    # Child Class
    """
    metric : F1 score
    """
    def __init__(self, metric: str='f1_score', learner: str='auto', ensemble: str='voting'):
        super().__init__(metric, learner, ensemble)
        # 'classification' Type
        self.type_ = 'classification'
        self.param['lgbm'] = {'objective': 'binary',
                              'learning_rate': 0.05,
                              'random_state': 42}   # LightGBM

class Regressor(Ensemble):
    # Child Class
    """
    metric : R-squared score
    """
    def __init__(self, metric: str='r2_score', learner: str='auto', ensemble: str='voting'):
        super().__init__(metric, learner, ensemble)
        # 'regression' Type
        self.type_ = 'regression'
