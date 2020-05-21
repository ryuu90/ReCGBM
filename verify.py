import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import matthews_corrcoef 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV 
import lightgbm as lgb
import sklearn.cluster
import Levenshtein

def ap_verification(test_set, test_seq, ap_centers):
    lev_similarity = -1*np.array([[Levenshtein.distance(w1,w2) for w1 in ap_centers['cleavage'].values] for w2 in test_seq['cleavage'].values])
    idx_class = np.argmax(lev_similarity, axis=1)
    idx_cluster = ap_centers['cluster'].values[idx_class]
    
    if idx_cluster.tolist() == test_set['cluster'].values.tolist():
        print("They are equal!")
    else:
        print("They are not Equal!")
    
    return 0

def model_test(lgb_model, test_set):
    x_test, y_test =  test_set.iloc[:,:-1], test_set.iloc[:,-1]
    
    y_pred = lgb_model.predict(x_test)
    
    y_round = np.around(y_pred)
    
    cm_test= confusion_matrix(y_test, y_round)
    total_test=sum(sum(cm_test))
    
    accuracy_test=(cm_test[0,0]+cm_test[1,1])/total_test
    
    sensitivity_test = cm_test[0,0]/(cm_test[0,0]+cm_test[0,1])
    
    specificity_test = cm_test[1,1]/(cm_test[1,0]+cm_test[1,1])

    MCC = matthews_corrcoef(y_test, y_round)
    return [accuracy_test, sensitivity_test, specificity_test, MCC]

def model_load(idx, arm, struct):
    model = lgb.Booster(model_file='./model_data/lgb_'+ arm + '_' + struct + '_' + str(idx) + '.txt')
    test_set = pd.read_pickle('./model_data/df_test_' + arm + '_' + struct + '_' + str(idx) +  '.pkl')
    test_seq = pd.read_pickle('./model_data/df_test_seq_' + arm + '_' + struct + '_' + str(idx) +  '.pkl')
    ap_centers = pd.read_pickle('./model_data/df_ap_' + arm + '_' + struct + '_' + str(idx) +  '.pkl')
    
    #x_test, y_test =  test_set.iloc[:,:-1], test_set.iloc[:,-1]
    return model, test_set, test_seq, ap_centers

def metric_print(metric_list):
    for idx, item in enumerate(metric_list):
        if idx%4 == 0:
            print("accuracy" + ": ", item)
        elif idx%4 == 1:
            print("sensitivity" + ": ", item)
        elif idx%4 == 2:
            print("specificity" + ": ", item)
        else:
            print("MCC" + ": ", item)

for i in range(10):
    print('round------' + str(i))

    print('quickfold_5p:')
    lgb_5p_qf, test_set, test_seq, ap_centers = model_load(i, '5p', 'qf')
    #ap_verification(test_set, test_seq, ap_centers)
    metric_5p_qf = model_test(lgb_5p_qf, test_set)
    metric_print(metric_5p_qf)
        
    print('quickfold_3p:')
    lgb_3p_qf, test_set, test_seq, ap_centers = model_load(i, '3p', 'qf')
    #ap_verification(test_set, test_seq, ap_centers)
    metric_3p_qf = model_test(lgb_3p_qf, test_set)
    metric_print(metric_3p_qf)
    
    print('RNAFold_5p:')
    lgb_5p_rf, test_set, test_seq, ap_centers = model_load(i, '5p', 'rf')
    #ap_verification(test_set, test_seq, ap_centers)
    metric_5p_rf = model_test(lgb_5p_rf, test_set)
    metric_print(metric_5p_rf)
    
    print('RNAFold_3p:')
    lgb_3p_rf, test_set, test_seq, ap_centers = model_load(i, '3p', 'rf')
    #ap_verification(test_set, test_seq, ap_centers)
    metric_3p_rf = model_test(lgb_3p_rf, test_set)
    metric_print(metric_3p_rf)

