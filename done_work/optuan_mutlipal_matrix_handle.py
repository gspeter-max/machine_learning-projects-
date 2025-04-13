
 # NSGAIIsampler
import numpy as np
from optuna.samplers import NSGAIISampler # that sampler is allow to sampler ( that choice hyperparater and pass to objective function ) to handle hyperparameter for multipal  matrix 
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score , precision_recall_curve, confusion_matrix ,f1_score


x, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5,
                           n_clusters_per_class=2, weights=[0.7, 0.3], flip_y=0.05, random_state=42)

def obejective(trials):
    params = {
            'n_estimators': trials.suggest_int('n_estimators',140, 200),
            'max_depth' : trials.suggest_int('max_depth',5,10),
            'min_samples_split' : trials.suggest_int('min_samples_split', 4,10),
            'min_samples_leaf': trials.suggest_int('min_samples_leaf', 2,10),
            'random_state' : 42,
            'class_weight': trials.suggest_categorical('class_weight',[None, 'balanced']),
        }
    model = RandomForestClassifier(**params)

    roc = cross_val_score(estimator = model,X = x, y= y, scoring ='roc_auc', cv =3 )
    f1 = cross_val_score(estimator = model, X = x, y =y , scoring = 'f1', cv =3)

    return np.mean(roc).astype(float),np.mean(f1).astype(float)

sampling = NSGAIISampler(seed = 42)
study = optuna.create_study(directions = ['maximize','maximize'] , sampler  = sampling )

study.optimize(obejective, n_trials = 20)
index = 0
value = 0
param = {}
for index, trial in enumerate(study.best_trials):
    values = trial.values[1]
    if values >=  value:
        print(values)
        param = trial.params

print(param)
best_model = RandomForestClassifier(**param)

sss = StratifiedShuffleSplit(n_splits= 1, random_state =42, test_size = 0.2)
train_index, test_index = next(sss.split(x,y))
x_train , x_test = x[train_index], x[test_index]
y_train , y_test = y[train_index], y[test_index]

best_model.fit(x_train , y_train )
y_proba  = best_model.predict_proba(x_test)[:,1]
precision, recall ,threshold = precision_recall_curve(y_test, y_proba)
f1 = 2* (precision * recall) / (precision - recall)
threshold_index = np.argmax(f1)
best_threshold = threshold[threshold_index]

y_pred = ( y_proba > best_threshold ).astype(int)

print(f' f1 score : {f1_score(y_test, y_pred)}')
print(f'confusion matrix : {confusion_matrix(y_test, y_pred)}')


