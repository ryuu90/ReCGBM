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

def categorical_feature(df_all, arm):
    categorical_feature = [] 
    for item, item_comp in zip(df_all["cleavage_" + arm].values, df_all["cleavage_"+ arm +"_comp"].values):
        temp = []
        for alph, alph_comp in zip(item, item_comp):
            temp.append(alph+alph_comp)
        categorical_feature.append(temp)
    df_temp = pd.DataFrame.from_records(categorical_feature)
    df_temp.columns = ['p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 'p_7', 'p_8', 'p_9', 'p_10', 'p_11', 'p_12', 'p_13', 'p_14']
    return df_temp

def lgbm(x_train_rel, y_train_rel, x_test_rel, y_test_rel):
    parameters = {
             "max_depth": [10, 20, 30, 40, 50, 60],
              "learning_rate" : [0.05, 0.1, 0.15],
              "num_leaves": [200, 300, 400],

    }
    
    gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                             objective = 'binary',
                             metric = 'binary_logloss',
                             verbose = -1)

    grid_xgb = GridSearchCV(gbm, param_grid=parameters, cv=5)
    grid_xgb.fit(x_train_rel, y_train_rel)
    clf = grid_xgb.best_estimator_
    
    
    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, x_train_rel.columns)), columns=['Value','Feature'])
    
    y_pred_rel = grid_xgb.predict(x_test_rel)
    
    cm_rel= confusion_matrix(y_test_rel,y_pred_rel)
    total_rel=sum(sum(cm_rel))
    
    accuracy_rel=(cm_rel[0,0]+cm_rel[1,1])/total_rel
    
    sensitivity_rel = cm_rel[0,0]/(cm_rel[0,0]+cm_rel[0,1])
    
    specificity_rel = cm_rel[1,1]/(cm_rel[1,0]+cm_rel[1,1])

    MCC = matthews_corrcoef(y_test_rel, y_pred_rel)
    return [accuracy_rel, sensitivity_rel, specificity_rel, MCC], feature_imp

def train_round(arm, chosen_idx, df_cluster):
    #ap_cluster
    df_origin = df_cluster["cleavage_" + arm]
    df_comp = df_cluster["cleavage_" + arm + "_comp"]
    seq = df_origin + df_comp
    df_seq = pd.DataFrame(seq.values, columns=['cleavage'])
    test_df = seq.index.isin(chosen_idx)
    
    df_train_cluster = pd.DataFrame.from_records(df_cluster.iloc[~test_df])
    df_test_cluster = pd.DataFrame.from_records(df_cluster.iloc[chosen_idx])
    
    df_train = pd.DataFrame.from_records(df_seq.loc[~test_df])
    df_test = pd.DataFrame.from_records(df_seq.loc[chosen_idx])
    
    lev_similarity = -1*np.array([[Levenshtein.distance(w1,w2) for w1 in df_train['cleavage'].values] for w2 in df_train['cleavage'].values])
    af = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
    af.fit(lev_similarity)
    cluster_centers_indices = af.cluster_centers_indices_
    
    df_train_cluster['cluster'] = af.labels_
    df_train_cat = categorical_feature(df_train_cluster, arm)
    df_train_cat['cluster'] = af.labels_
    df_train_cat['label'] = df_train_cluster['label'].values
    
    lev_similarity = -1*np.array([[Levenshtein.distance(w1,w2) for w1 in df_train['cleavage'].values[cluster_centers_indices]] for w2 in df_test['cleavage'].values])
    idx_class = np.argmax(lev_similarity, axis=1)
    idx_cluster = cluster_centers_indices[idx_class]
    df_test_cat = categorical_feature(df_test_cluster, arm)
    df_test_cat['cluster'] = df_train_cluster.iloc[idx_cluster]['cluster'].values
    df_test_cat['label'] = df_test_cluster['label'].values
    
    Label_Enc_list =['cluster', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6', 'p_7', 'p_8', 'p_9', 'p_10', 'p_11', 'p_12', 'p_13', 'p_14']
    for i in Label_Enc_list:
        df_train_cat[i] = df_train_cat[i].astype('category')
        df_test_cat[i] = df_test_cat[i].astype('category')
    
    df_test_cluster_, df_train_cluster_ = shuffle(df_test_cat), shuffle(df_train_cat)
    x_train_cluster, y_train_cluster, x_test_cluster, y_test_cluster = df_train_cluster_.iloc[:,:-1], df_train_cluster_.iloc[:,-1], df_test_cluster_.iloc[:,:-1], df_test_cluster_.iloc[:,-1]
    
    metric_cat, feature_imp_cat = lgbm(x_train_cluster, y_train_cluster, x_test_cluster, y_test_cluster)
    for idx, item in enumerate(metric_cat):
        if idx%4 == 0:
            print("accuracy" + ": ", item)
        elif idx%4 == 1:
            print("sensitivity" + ": ", item)
        elif idx%4 == 2:
            print("specificity" + ": ", item)
        else:
            print("MCC" + ": ", item)


df_all_qf = pd.read_pickle("cleav.pkl")
df_all_rf = pd.read_pickle("cleav_rf.pkl")
df_pair_cat_5p = pd.DataFrame()
df_pair_cat_3p = pd.DataFrame()
df_pair_cat_5p_r = pd.DataFrame()
df_pair_cat_3p_r = pd.DataFrame()
df_pair_cat_5p['cleavage_5p'] = df_all_qf['cleavage_5p']
df_pair_cat_5p[ 'cleavage_5p_comp'] = df_all_qf[ 'cleavage_5p_comp']
df_pair_cat_5p[ 'label'] = df_all_qf[ 'label']
df_pair_cat_3p['cleavage_3p'] = df_all_qf['cleavage_3p']
df_pair_cat_3p[ 'cleavage_3p_comp'] = df_all_qf[ 'cleavage_3p_comp']
df_pair_cat_3p['label'] = df_all_qf[ 'label']
df_pair_cat_5p_r['cleavage_5p'] = df_all_rf['cleavage_5p']
df_pair_cat_5p_r[ 'cleavage_5p_comp'] = df_all_rf[ 'cleavage_5p_comp']
df_pair_cat_5p_r['label'] = df_all_rf['label']
df_pair_cat_3p_r['cleavage_3p'] = df_all_rf['cleavage_3p']
df_pair_cat_3p_r['cleavage_3p_comp'] = df_all_rf['cleavage_3p_comp']
df_pair_cat_3p_r['label'] = df_all_rf['label']     

chosen_idx = np.load('./example/test_idx.npy')   

print('quickfold_5p:')
train_round('5p', chosen_idx, df_pair_cat_5p)
print('quickfold_3p:')
train_round('3p', chosen_idx, df_pair_cat_3p)
print('RNAFold_5p:')
train_round('5p', chosen_idx, df_pair_cat_5p_r)
print('RNAFold_3p:')
train_round('3p', chosen_idx, df_pair_cat_3p_r)