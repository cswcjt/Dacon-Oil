# ---------------------------------------------------------------------------- #
#                 Make Model Pipeline for the multiple tests                   #
# ---------------------------------------------------------------------------- #
"""
<Module Version>
numpy              1.23.4
pandas             1.5.1
scikit-learn       1.1.3
xgboost            1.7.1
lightgbm           3.3.3
optuna             2.10.1
"""
# 
#
# ---------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
from typing import Optional, Union
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
    def __init__(self, metric: str,
                 objecitve: str,
                 learner: Union[str, list[str]]='auto',
                 ensemble: Optional[str]='voting',
                 learning_rate: Optional[float]=0.005,
                 random_state: Optional[int]=42,
                 early_stopping_rounds: Optional[int]=10,
                 optimize: bool=False,
                 n_trials: int=20,
                 cv: int=5,
                 N: int=5,
                 **kwargs: any):
        """
        metric : sklearn.metrics 내장함수 활용
        learner : 'auto' - rf, xgb, lgbm ensemble / 'rf' - RandomForest / 'xgb' - XGBoost / 'lgbm' - LightGBM
        ensemble : 'voting', 'stacking'
        """

        # 'classification' , 'regression'
        self.type_ = objecitve

        if type(learner) is list:
            self.learner_ = learner
        else:
            self.learner_ = ['rf', 'xgb', 'lgbm'] if learner == 'auto' else [learner]
        
        self.ensemble_ = ensemble if ensemble in ['voting', 'stacking'] else 'voting'
        self.metric_ = metric
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.optimize_ = optimize
        self.n_trials_ = n_trials
        self.cv_ = cv
        self.N_ = N

        if kwargs:
            for key, value in kwargs.items:
                if hasattr(self, key):
                    getattr(self, key, value)
                    if key == 'learner':
                        if type(learner) is list:
                            self.learner_ = learner
                        else:
                            self.learner_ = ['rf', 'xgb', 'lgbm'] if learner == 'auto' else [learner]

        
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
                    'random_state': self.random_state},
            
            'xgb' : {'learning_rate': self.learning_rate,
                     'nthread' : -1,
                     'n_jobs': -1,
                     'tree_method': 'gpu_hist',
                     'predictor': 'gpu_predictor',
                     'random_state': self.random_state},
            
            'lgbm' : {'learning_rate': self.learning_rate,
                      'n_jobs': -1,
                      'random_state': self.random_state}
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

    def make_weights(self, n_learners: int, N: int) -> list:
        # x+y+z = N인 음이 아닌 정수 (x, y, z) 순서쌍 만들기
        weights = []

        if n_learners == 3:
            for i in range(N+1):
                for j in range(N+1-i):
                    k = N-i-j
                    temp = [i/N, j/N, k/N]
                    weights.append(temp)
        elif n_learners == 2:
            for i in range(N+1):
                j = N-i
                temp = [i/N, j/N]
                weights.append(temp)

        return weights

    def model_fit(self, model: callable,
                  learner: str,
                  X_train: pd.DataFrame, 
                  y_train: Union[pd.Series, pd.DataFrame, np.ndarray],
                  X_val: Optional[pd.DataFrame],
                  y_val: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]]) -> None:
            
            if learner == 'rf':
                getattr(model, 'fit')(X_train, y_train)
            else:
                getattr(model, 'fit')(X_train,
                                      y_train,
                                      eval_set=[(X_val, y_val)],
                                      early_stopping_rounds=self.early_stopping_rounds,
                                      verbose=True)

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> None:

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        for learner in self.learner_:
            if self.optimize_:
                # RF, XGB, LGBM 순서대로 hyper-parameter tuning
                param = self.optimizer(X_train, y_train, learner, self.n_trials_, self.cv_)
                # Hyper-parameter fix + tuning
                self.param[learner].update(param)
            
            # Initialize final models
            self.models[learner] = self.learners[self.type_][learner](**self.param[learner])

        # If only 1 model is used, final model is just fitted model
        if len(self.learner_) < 2:
            self.final_ensemble = self.models[self.learner_[0]]
            self.model_fit(self.final_ensemble, self.learner_[0],
                           X_train, y_train,
                           X_val, y_val)
        # Else if, fit ensemble model
        else:
            estimators = [(learner, self.models[learner]) for learner in self.learner_]

            ensemble_param = {
                'estimators': estimators,
                'n_jobs': -1
            }

            if self.ensemble_ == 'voting':
                if self.type_ == 'classification':
                    ensemble_param.update({'voting': 'soft'})

            elif self.ensemble_ == 'stacking':
                ensemble_param.update({'cv': self.cv_,
                                       'final_estimator': self.learners[self.type_]['lgbm']()})
            
            self.final_ensemble = self.voters[self.type_][self.ensemble_](**ensemble_param)

            if self.optimize_ and (self.ensemble_ == 'voting'):
                # 'weights': weights
                weights = self.make_weights(n_learners=len(self.learner_), N=self.N_)
                grid_params = {'weights': weights}
                grid_Search = GridSearchCV(param_grid=grid_params, estimator=self.final_ensemble, scoring=self.metric_dict[self.type_][self.metric_])
                grid_Search.fit(X_train, y_train)
                self.final_ensemble = grid_Search.best_estimator_
            
            getattr(self.final_ensemble, 'fit')(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return getattr(self.final_ensemble, 'predict')(X_test)

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.metric_ in ['rmse', 'rmsle']:
            return self.metric_dict[self.type_][self.metric_[1:]](y_true, y_pred, squared=False)
        else:
            return self.metric_dict[self.type_][self.metric_](y_true, y_pred)

    def K_fold(self, model: callable,
               learner: str,
               X: pd.DataFrame,
               y: Union[pd.Series, np.ndarray],
               cv: int) -> list:
        scores = []
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        try:
            for train_idx, val_idx in folds.split(X, y):
                continue
        except:
            folds = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        if cv == 1:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
            self.model_fit(model, learner,
                           X_train, y_train,
                           X_val, y_val)
            score = self.score(y_val, model.predict(X_val))
            scores.append(score)

        else:
            for train_idx, val_idx in folds.split(X, y):
                X_train = X.iloc[train_idx, :]
                y_train = y.iloc[train_idx]
                
                X_val = X.iloc[val_idx, :]
                y_val = y.iloc[val_idx]
                
                self.model_fit(model, learner,
                           X_train, y_train,
                           X_val, y_val)
                score = self.score(y_val, model.predict(X_val))
                scores.append(score)

        return scores

    def objective(self, trial: Trial,
                  X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
                  learner: str, cv: int) -> float:
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
        scores = self.K_fold(model, learner, X, y, cv)

        return np.mean(scores)
    
    def optimizer(self, X: pd.DataFrame,
                  y: Union[pd.Series, np.ndarray], learner: str,
                  n_trials: int, cv: int) -> dict:
        
        direction = self.metric_direction_dict[self.type_][self.metric_]
        study = optuna.create_study(direction=direction, sampler=TPESampler())
        study.optimize(lambda trial : self.objective(trial, X, y, learner, cv),
                       n_trials=n_trials)
        print('Best trial: score {},\nparams: {}'.format(study.best_trial.value, study.best_trial.params))
        return study.best_trial.params

class BinaryCalssifier(Ensemble):
    # Child Class
    """
    metric : F1 score
    """
    def __init__(self, metric: str='f1_score',
                 objecitve: str='classification',
                 learner: Union[str, list[str]]='auto',
                 ensemble: Optional[str]='voting',
                 learning_rate: Optional[float]=0.005,
                 random_state: Optional[int]=42,
                 early_stopping_rounds: Optional[int]=10,
                 optimize: bool=False,
                 n_trials: int=20,
                 cv: int=5,
                 N: int=5,
                 **kwargs: any):

        super().__init__(metric, objecitve,
                         learner, ensemble,
                         learning_rate, random_state,
                         early_stopping_rounds,
                         optimize, n_trials,
                         cv, N, **kwargs)

class Regressor(Ensemble):
    # Child Class
    """
    metric : R-squared score
    """
    def __init__(self, metric: str='r2_score',
                 objecitve: str='regression',
                 learner: Union[str, list[str]]='auto',
                 ensemble: Optional[str]='voting',
                 learning_rate: Optional[float]=0.005,
                 random_state: Optional[int]=42,
                 early_stopping_rounds: Optional[int]=10,
                 optimize: bool=False,
                 n_trials: int=20,
                 cv: int=5,
                 N: int=5,
                 **kwargs: any):

        super().__init__(metric, objecitve,
                         learner, ensemble,
                         learning_rate, random_state,
                         early_stopping_rounds,
                         optimize, n_trials,
                         cv, N, **kwargs)