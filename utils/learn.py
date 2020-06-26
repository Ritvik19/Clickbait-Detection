import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm


from sklearn.metrics import log_loss, roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from imblearn.over_sampling import SMOTE

def train_classifier(model, feats, target, feat_names=None, labels=None, interpret=None, loss_func=log_loss, k=11, **kwargs):
    model_performance = {
        'loss': [],
        'score': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1 score': []
    }
    cm = []
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=101)
    if 'smote' in kwargs.keys() and kwargs['smote'] == True:
        smote = SMOTE()

    for train_indices, test_indices in tqdm(skf.split(feats, target)):
        X_train = feats[train_indices]
        y_train = target[train_indices]

        if 'smote' in kwargs.keys() and kwargs['smote'] == True:
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        X_test = feats[test_indices]
        y_test = target[test_indices]

        model.fit(X_train, y_train)
        y_pred_ = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        model_performance['loss'].append(loss_func(y_test, y_pred_))
        model_performance['accuracy'].append(accuracy_score(y_test, y_pred))
        model_performance['score'].append(roc_auc_score(y_test, y_pred_[:, 1].reshape((-1,))))
        model_performance['precision'].append(precision_score(y_test, y_pred))
        model_performance['recall'].append(recall_score(y_test, y_pred))
        model_performance['f1 score'].append(f1_score(y_test, y_pred))
        cm.append(normalize(confusion_matrix(y_test, y_pred), axis=1, norm='l1') * 100)


    fig = plt.figure(figsize=(20, 6))

    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    
    if 'plot_loss' in kwargs.keys() and kwargs['plot_loss'] == True:
        ax1.plot(model_performance['loss'], label='loss per iteration')
        ax1.plot(np.ones(k)*np.mean(model_performance['loss']), '--', label='mean loss')

    ax1.plot(model_performance['accuracy'], label='accuracy per iteration')
    ax1.plot(np.ones(k)*np.mean(model_performance['accuracy']), '--', label='mean accuracy')

    if 'plot_roc_auc' in kwargs.keys() and kwargs['plot_roc_auc'] == True:
        ax1.plot(model_performance['score'], label='score per iteration')
        ax1.plot(np.ones(k)*np.mean(model_performance['score']), '--', label='mean score')
    
    if 'plot_precision' in kwargs.keys() and kwargs['plot_precision'] == True:
        ax1.plot(model_performance['precision'], label='precision per iteration')
        ax1.plot(np.ones(k)*np.mean(model_performance['precision']), '--', label='mean precision')    

    if 'plot_recall' in kwargs.keys() and kwargs['plot_recall'] == True:
        ax1.plot(model_performance['recall'], label='recall per iteration')
        ax1.plot(np.ones(k)*np.mean(model_performance['recall']), '--', label='mean recall')    

    if 'plot_f1' in kwargs.keys() and kwargs['plot_f1'] == True:
        ax1.plot(model_performance['f1 score'], label='f1 score per iteration')
        ax1.plot(np.ones(k)*np.mean(model_performance['f1 score']), '--', label='mean f1 score')    

    ax1.grid()
    ax1.legend()
    ax1.set_xlabel('fold')
    ax1.set_ylabel('value')
    ax1.set_title('Model Performance')

    ax2 = plt.subplot2grid((1, 3), (0, 2), colspan=1)
    cm = np.mean(cm, axis=0)
    sns.heatmap(cm, annot=True, square=True, ax=ax2, cmap='Blues')
    ax2.set_title('Confusion Matrix')    
    if labels is not None:
        ax2.set_xticklabels(labels, rotation = 45)
        ax2.set_yticklabels(labels, rotation = 45)
    
    
    if interpret is not None:
        if feat_names is None:
            assert "Provide feat_names"
        fig, ax = plt.subplots(figsize=(20, 6))
        if interpret == 'linear':
            ax.bar(x=feat_names+['intercept'], height=np.append(model.coef_[0], model.intercept_[0]))
            ax.grid()
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
            ax.set_title('Model Coefficients')
        if interpret == 'tree':
            ax.bar(x=feat_names, height=model.feature_importances_)
            ax.grid()
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
            ax.set_title('Model Coefficients')            
    
    return model_performance, cm, model