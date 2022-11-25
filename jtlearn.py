import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional, Union

from collections import Counter # 샘플결과 확인
from sklearn.model_selection import train_test_split # 트테트테
from sklearn.decomposition import PCA # 차원축소
from sklearn.ensemble import RandomForestClassifier # 모델선택
from sklearn.metrics import f1_score # 성과지표
from sklearn.metrics import classification_report # 성과지표
from imblearn.under_sampling import * # 임벨런스
from imblearn.over_sampling import * # 임벨런스
from imblearn.combine import * # 임벨런스
from imblearn.pipeline import Pipeline # 파이프라인구축

from lightgbm.sklearn import LGBMClassifier
from xgboost.sklearn import XGBClassifier

class Preprocessing:
    def __init__(self,
                 learner: Optional[str]=None,
                 objective: Optional[str]=None,
                 sampler: Optional[str]=None,
                 test_size: Optional[Union[int, float]]=0.1,
                 group_split_feature: Optional[str]=None,
                 random_state: Optional[str]=42,
                 dimensionality: Optional[Union[str, callable]]='pca',
                 **kwargs: any,
                 ):
        """
        categorical_feature: str="COMPONENT_ARBITRARY", 
        test_size:int=0.1, random_state_: int=42 ,dimensionality: str="pca"
        """

        # preprocessing for data set
        self.learner = learner
        self.objective = objective
        self.sampler = sampler
        self.test_size = test_size
        self.group_split_feature = group_split_feature
        self.random_state = random_state
        self.dimensionality = dimensionality
        
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        # preprocessing for learning model
        # self.learners_dict[self.learner]
        self.learners_dict = {
            'rf': RandomForestClassifier,
            'xgb': XGBClassifier,
            'lgbm': LGBMClassifier
        }

        # preprocessing for sampling model
        # self.samplers_dict[self.objective][self.sampler]
        self.samplers_dict = {
            "under": {
                'RandomUnderSampler': RandomUnderSampler,
                'TomekLinks': TomekLinks,
                'CondensedNearestNeighbour': CondensedNearestNeighbour, 
                'OneSidedSelection': OneSidedSelection,
                'EditedNearestNeighbours': EditedNearestNeighbours,
                'NeighbourhoodCleaningRule': NeighbourhoodCleaningRule
            },

            "over": {
                'RandomOverSampler': RandomOverSampler,
                'SMOTE': SMOTE,
                'ADASYN': ADASYN,
                'NeighbourhoodCleaningRule': NeighbourhoodCleaningRule
            },

            "hybrid": {
                'SMOTEENN': SMOTEENN,
                'SMOTETomek': SMOTETomek
            }
        }

        # create new attrubutes for methods
        self.my_learner = self.learners_dict[self.learner]
        self.my_sampler = self.samplers_dict[self.objective][self.sampler]

    # def __init__(self, **kwargs):
    #     """
    #     categorical_feature: str="COMPONENT_ARBITRARY", 
    #     test_size:int=0.1, random_state_: int=42 ,dimensionality: callable= PCA()
    #     """

    #     # preprocessing for data set
    #     self.categorical_feature = kwargs["categorical_feature"]

    #     # self.concat_df = pd.concat([X, y], axis=1) 
    #     self.test_size = kwargs["test_size"]

    #     # preprocessing for learning model
    #     self.learners_dict = {
    #         'classification': {
    #             'RF': RandomForestClassifier,
    #             'XGB': XGBClassifier,
    #             'LGBM': LGBMClassifier
    #         },
        
    #         'regression': {
    #             'RF': RandomForestRegressor,
    #             'XGB': XGBRegressor,
    #             'LGBM': LGBMRegressor
    #         }
    #     }

    #     # preprocessing for sampling model
    #     self.samplers_dict = {
    #         "under": {
    #             'RandomUnderSampler': RandomUnderSampler,
    #             'TomekLinks': TomekLinks,
    #             'CondensedNearestNeighbour': CondensedNearestNeighbour, 
    #             'OneSidedSelection': OneSidedSelection,
    #             'EditedNearestNeighbours': EditedNearestNeighbours,
    #             'NeighbourhoodCleaningRule': NeighbourhoodCleaningRule
    #         },

    #         "over": {
    #             'RandomOverSampler': RandomOverSampler,
    #             'ADASYN': ADASYN,
    #             'NeighbourhoodCleaningRule': NeighbourhoodCleaningRule
    #         },

    #         "hybrid": {
    #             'SMOTEENN': SMOTEENN,
    #             'SMOTETomek': SMOTETomek
    #         }
    #     }

    #     # preprocessing for dimensionality
    #     self.dimensionality = kwargs["dimensionality"]

    #     # create new attrubutes for methods 
    #     learner = kwargs["learner"]
    #     sampler = kwargs["sampler"]
    #     self.my_learner = self.learners_dict[learner[0]][learner[1]]
    #     self.my_sampler = self.samplers_dict[sampler[0]][sampler[1]]
    #     self.random_state_ = kwargs["random_state_"]

    def sampling(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> tuple:
        try: 
            sampler = self.my_sampler(random_state=self.random_state, n_jobs=-1)
            X2, y2 = sampler.fit_resample(X, y)
        
        except ValueError:
            print("categorical value 넣지마세요!")

        except TypeError: 
            print("random_state 없는 샘플러")
            sampler = self.my_sampler()
            X2, y2 = sampler.fit_resample(X, y)
        
        
        print(f"{self.sampler} completed resampling X and y" )
        return X2, y2

    def sampling_group(self, X: pd.DataFrame,
                       y: Union[pd.Series, np.ndarray],
                       categorical_feature: str) -> dict: 
        """
        divide train_df to make each group df
        return grouped df list 
        """

        # concat X2 and y2 to divide groups
        if hasattr(y, 'name'):
            y_col = y.name
        else:
            y_col = 'Y_LABEL'

        concat_df: pd.DataFrame = pd.concat([X,y], axis=1)
        group_dic = {}

        for criteria in sorted(concat_df[categorical_feature].unique()): 
            print(f"dividing my df on {criteria}")
            temp_df = concat_df.loc[concat_df[categorical_feature] == criteria,].drop(columns=categorical_feature)

            # make grouped X, y
            X2 = temp_df.drop(columns=[y_col])
            y2 = temp_df[y_col]
            
            # imbalance sampling
            X3, y3 = self.sampling(X2, y2)
            group_dic.update({criteria: (X3, y3)})
        
        return group_dic

    def split_X_y_bundle(self, X_y_bundle: Union[tuple, dict]) -> Union[tuple, dict]: 
        """
        split train and validation data set
        return X_train, X_val, y_train, y_val
        """

        if type(X_y_bundle) == tuple: 
            (X, y) = X_y_bundle
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            return X_train, X_val, y_train, y_val

        else: 
            split_dict = {}
            for key, (X, y) in X_y_bundle.items():
                (X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
                split_dict.update({key : (X_train, X_val, y_train, y_val)})     
            return split_dict

    def feature_importance_for_groups(self, split_dict: dict[Union[int, str], tuple]) -> dict[str, pd.Series]: # -> 메서드 가지고 오면 피처임포턴스 리턴하는 메서드
        # nan 값 처리 후 사용 가능
        classifier = self.my_learner()
        score_dict = {}
        feature_importance_dict = {}

        for criteria, (X_train, X_val, y_train, y_val) in split_dict.items(): 
            classifier.fit(X_train, y_train) # Random Forest 학습을 위해 parameter 채우기
            pred = classifier.predict(X_val) # Random Forest 테스트를 위해 parameter 채우기
            score = f1_score(y_val, pred)  # f1_score 계산
            score_dict.update({criteria : score})
            print("f1_score : %.3f" % score)

            importances = classifier.feature_importances_
            ftr_importances = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
            feature_importance_dict.update({criteria : ftr_importances})

        return feature_importance_dict

    def choose_drop_features(self, feature_importance_dict: dict[str, pd.Series],
                            threshold: int=0, draw: bool=False) -> dict[Union[int, str], list]:
        """
        return list of all feature_importance for each group 
        """

        if draw == True:
            for criteria, feature_importance in feature_importance_dict.items(): 
                plt.figure(figsize=(12, 6))
                plt.title(f'{criteria} Feature Importances')
                sns.barplot(x=feature_importance, y=feature_importance.index)
                plt.show()

        drop_target_dict = {}
        
        for criteria, feature_importance in feature_importance_dict.items():
            temp_df: pd.DataFrame = feature_importance.reset_index()
            temp_df.columns = ["name", "value"]
            
            drop_target_dict[criteria] = temp_df[temp_df.value <= threshold].name.to_list()

        return drop_target_dict

    def print_report(self, split_dict: dict[str, tuple]) -> str: 
        sampler = self.my_sampler(random_state=self.random_state)
        print(sampler)
        classifier = self.my_learner()
        print(classifier)
        dimensionality = self.dimensionality()
        print(type(dimensionality))
        # sampling method, dimensionality, model
        pipeline = Pipeline([('sampling_method', sampler), ('dimensionality', dimensionality), ('model', classifier)]) 
        
        for _, (X_train, X_val, y_train, y_val) in split_dict.items(): 
            pipeline.fit(X_train, y_train) 
            y_hat = pipeline.predict(X_val)
            print(f"{dimensionality} 사용한 pipe line")
            print(classification_report(y_val, y_hat))