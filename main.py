"""
<Module Version>
numpy              1.23.4
pandas             1.5.1
scikit-learn       1.1.3
imbalanced-learn   0.9.1
xgboost            1.7.1
lightgbm           3.3.3
optuna             2.10.1
tqdm               4.64.1
"""

from sklearn.decomposition import PCA # 차원축소
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score

from imblearn.under_sampling import * # 임벨런스
from imblearn.over_sampling import * # 임벨런스
from imblearn.combine import * # 임벨런스

import os
import copy
import pandas as pd
import matplotlib.pyplot as plt

from jtlearn import Preprocessing
from ensemble import BinaryCalssifier, Regressor

# Load Data
save_path = 'submission/'
base_path = 'data/'

train = pd.read_csv(base_path + 'train.csv')
test = pd.read_csv(base_path + 'test.csv')
submission = pd.read_csv(base_path + 'sample_submission.csv')

# Preprocessing

train = train.fillna(0)

X = train.drop(columns=["ID", "Y_LABEL"])
y = train["Y_LABEL"]
test = test.drop(columns=['ID'])

train_num_cols = X.drop(columns=['COMPONENT_ARBITRARY']).columns.tolist()
test_num_cols = test.drop(columns=['COMPONENT_ARBITRARY']).columns.tolist()

ss = StandardScaler()
ss2 = StandardScaler()

ss2.fit(X[test_num_cols])
X[train_num_cols] = ss.fit_transform(X[train_num_cols])
test[test_num_cols] = ss2.transform(test[test_num_cols])

X.COMPONENT_ARBITRARY = X.COMPONENT_ARBITRARY.map({"COMPONENT1" : 1, "COMPONENT2" : 2, "COMPONENT3" : 3, "COMPONENT4" : 4})
test.COMPONENT_ARBITRARY = test.COMPONENT_ARBITRARY.map({"COMPONENT1" : 1, "COMPONENT2" : 2, "COMPONENT3" : 3, "COMPONENT4" : 4})

_, X_test, _, y_test = train_test_split(X, y, test_size=len(test), random_state=6974)
X_test: pd.DataFrame = X_test.reset_index(drop=True)
X_test_reg_valid = X_test.drop(columns=test.columns)
X_test = X_test[test.columns]

# ("under", "RandomUnderSampler")
# ("over", "RandomOverSampler")
# ("hybrid", "SMOTEENN")
# sampler
variable_dict = {
    "test_size": 0.1, 
    "learner": 'xgb',
    "objective": "hybrid",
    "sampler": "SMOTEENN",
    "random_state_": 42,
    "dimensionality": PCA
}

balancing = Preprocessing(**variable_dict)
print()

# 샘플링 그룹핑 스플릿
grouped_dict = balancing.sampling_group(X, y, categorical_feature='COMPONENT_ARBITRARY')
split_X_y_bundle = balancing.split_X_y_bundle(grouped_dict)
print()


# 피처임포턴스 확인
features = balancing.feature_importance_for_groups(split_X_y_bundle)
drop_target_list = balancing.chose_drop_features(features, draw=False)
print()
print(drop_target_list)
print()

drop_list_reg = list(set(drop_target_list) - set(test.columns))
drop_list_test = list(set(drop_target_list) & set(test.columns))
X2 = X.drop(columns=drop_list_reg)

resid_cols = X2.drop(columns=test.columns).columns

X_reg = X[test.columns]


# f = open("C:/doit/새파일.txt", 'w')
# for i in range(1, 11):
#     data = "%d번째 줄입니다.\n" % i
#     f.write(data)
# f.close()

reg_dict = {}
"""
regressor hyper parameters
{
    "metric": , # 평가지표(default='r2_score')
    "learner": , # 학습모델['rf', 'xgb', 'lgbm'] 중 아무 조합 선택 ㄱㄱ(default=['rf', 'xgb', 'lgbm'])
    "ensemble": , # ['voting', 'stacking'](default='voting')
    "learning_rate": , # 학습률(default=0.05)
    "random_state": , # 난수 seed(default=42)
    "early_stopping_rounds": , # overfitting 방지용(default=10)
    "optimize": , # optuna 사용할지 말지 True or False 사용 ㄱㄱ(default=False)
}
reg.fit parameters
{
    "n_trials": , # optuna 횟수(default=20)
    "cv": , # K-fold CV의 K(default=5)
    "N": , # voting에서 모델별 weights의 조합가지수(default=5)
}
"""
for col in resid_cols:
    reg = Regressor(optimize=True)
    y_reg = X[col]
    y_reg_test = X_test_reg_valid[col]
    # train-validation split
    X_reg_train, X_reg_val, y_reg_train, y_reg_val = train_test_split(X_reg, y_reg, test_size=0.2, random_state=69)

    # model fitting
    reg.fit(X_reg_train, y_reg_train, n_trials=10, cv=5, N=3) # n_trials: optuna 조지는 정도 / cv: K-fold의 K값

    # prediction
    y_reg_train_pred = reg.predict(X_reg_train)
    y_reg_val_pred = reg.predict(X_reg_val)
    y_reg_test_pred = reg.predict(X_test)
    
    # scoring
    score_train = reg.score(y_reg_train, y_reg_train_pred)
    score_val = reg.score(y_reg_val, y_reg_val_pred)
    score_test = reg.score(y_reg_test, y_reg_test_pred)
    reg_dict[col] = reg
    print(f"--------------- {col} ---------------")
    print(f"Model Information")
    print("Train R^2 score is %.4f" % (score_train))
    print("Validation R^2 score is %.4f" % (score_val))
    print("Test R^2 score is %.4f" % (score_test))
    print()

temp_test = copy.deepcopy(test)
temp_X_test = copy.deepcopy(X_test)

for col, reg in reg_dict.items():
    test[col] = reg.predict(temp_test)
    X_test[col] = reg.predict(temp_X_test)

X3 = X2.drop(columns=drop_list_test)
test = test.drop(columns=drop_list_test)
X_test = X_test.drop(columns=drop_list_test)

# Syncronize columns order between X3 and test
test = test[X3.columns]
X_test = X_test[X3.columns]

# 샘플링 그룹핑 스플릿
grouped_dict = balancing.sampling_group(X3, y, categorical_feature='COMPONENT_ARBITRARY')        
split_X_y_bundle = balancing.split_X_y_bundle(grouped_dict)

test_final = pd.DataFrame()
dummy_test_final = pd.DataFrame()

"""
classifier hyper parameters
{
    "metric": , # 평가지표(default='f1_score')
    "learner": , # 학습모델['rf', 'xgb', 'lgbm'] 중 아무 조합 선택 ㄱㄱ(default=['rf', 'xgb', 'lgbm'])
    "ensemble": , # ['voting', 'stacking'](default='voting')
    "learning_rate": , # 학습률(default=0.05)
    "random_state": , # 난수 seed(default=42)
    "early_stopping_rounds": , # overfitting 방지용(default=10)
    "optimize": , # optuna 사용할지 말지 True or False 사용 ㄱㄱ(default=False)
}
clf.fit parameters
{
    "n_trials": , # optuna 횟수(default=20)
    "cv": , # K-fold CV의 K(default=5)
    "N": , # voting에서 모델별 weights의 조합가지수(default=5)
}
"""
for criteria, (X_train, X_val, y_train, y_val) in split_X_y_bundle.items():
    test_temp = test[test.COMPONENT_ARBITRARY == criteria]
    test_temp = test_temp.drop(columns=['COMPONENT_ARBITRARY'])

    X_test_temp = X_test[X_test.COMPONENT_ARBITRARY == criteria]
    X_test_temp = X_test_temp.drop(columns=['COMPONENT_ARBITRARY'])
    
    # model initializing
    clf = BinaryCalssifier(learner='auto', optimize=True)

    # model training
    clf.fit(X_train, y_train, n_trials=20, cv=5) # n_trials: optuna 조지는 정도 / cv: K-fold의 K값

    # prediction
    y_train_pred = clf.predict(X_train)
    y_pred = clf.predict(X_val)
    
    # scoring
    score_train = clf.score(y_train, y_train_pred)
    score_val = clf.score(y_val, y_pred)
    print("Train F1_score is %.4f" % (score_train))
    print("Validation F1_score is %.4f" % (score_val))

    # fill prediction value in test data
    test_temp['Y_LABEL'] = clf.predict(test_temp)
    test_final = pd.concat([test_final, test_temp], axis=0)

    # fill prediction value in dummy test data
    X_test_temp['Y_LABEL'] = clf.predict(X_test_temp)
    dummy_test_final = pd.concat([dummy_test_final, X_test_temp], axis=0)

test_final = test_final.sort_index()
dummy_test_final = dummy_test_final.sort_index()
y_pred = dummy_test_final.Y_LABEL.values

"""
Only be able to operate in jupyter notebook enviornment
"""
conf_matrix = confusion_matrix(y_test, dummy_test_final.Y_LABEL.values)
# cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix,
#                                     display_labels = ['불량', '정상'])
# cm_display.plot()
# plt.show()


print('*' * 40)
print('Final model precision:\t%.4f' % (precision_score(y_test, y_pred)))
print('Final model recall:\t%.4f' % (recall_score(y_test, y_pred)))
print('Final model f1_score:\t%.4f' % (f1_score(y_test, y_pred)))
print('*' * 40)

if 'submission_oil.csv' in os.listdir(save_path):
    count = 0
    for name in os.listdir(save_path):
        if 'submission_oil' in name:
            count += 1
    filename = f"submission_oil{count + 1}.csv"
else:
    filename = 'submission_oil.csv'

# Export submission file
submission.Y_LABEL = test_final.Y_LABEL
submission.to_csv(save_path + filename, index=False)