from sklearn.metrics import recall_score, precision_score, f1_score, det_curve, accuracy_score
import pandas as pd
import numpy as np

def get_test_accuracy_of(model, test_data, test_labels):
    if "Decision" in str(type(model)):
        i = 1
    else: i = 0
    y_pred = model.predict(test_data)
    out = [1 if i > 0.5 else 0 for i in y_pred]
    fpr, fnr, _ = det_curve(test_labels, y_pred)
    metrics = {
        "Precision ↑ ": [precision_score(test_labels, out)],
        "Recall ↑ ": [recall_score(test_labels, out)],
        "F1 Score ↑ ": [f1_score(test_labels, out)],
        "FPR ↓ ": [fpr[i]],
        "FNR ↓ ": [fnr[i]],
        "Accuracy ↑ ": accuracy_score(test_labels, out)
    }
    df = pd.DataFrame.from_dict(metrics)
    return df

def NSL_KDD(base_path):
    train = pd.read_csv(base_path + "/NSLKDD/train.csv")
    test = pd.read_csv(base_path + "/NSLKDD/test.csv")

    train_labels = np.array([0 if i=='Normal' else 1 for i in train['label']])
    test_labels = np.array([0 if i=='Normal' else 1 for i in test['label']])
    train["label"] = train_labels
    scaled_train_data = train.drop(["label"],axis=1)
    scaled_test_data = test.drop(["label"],axis=1)

    return scaled_train_data, train_labels, scaled_test_data, test_labels

# UNSW
def UNSW(base_path):
    train = pd.read_csv(base_path + "/UNSW_NB15_training-set.csv")
    test = pd.read_csv(base_path + "/UNSW_NB15_testing-set.csv")

    train_labels = np.array([0 if i==1 else 1 for i in train['label']])
    test_labels = np.array([0 if i==1 else 1 for i in test['label']])
    train["label"] = train_labels
    scaled_train_data = train.drop(["label"],axis=1)
    scaled_test_data = test.drop(["label"],axis=1)

    return scaled_train_data, train_labels, scaled_test_data, test_labels

def AWID(base_path):
    train = pd.read_csv(base_path + "/AWID2/train.csv")
    test = pd.read_csv(base_path + "/AWID2/test.csv")

    train_labels = np.array([0 if i=='normal' else 1 for i in train['label']])
    test_labels = np.array([0 if i=='normal' else 1 for i in test['label']])
    train["label"] = train_labels
    scaled_train_data = train.drop(["label"],axis=1)
    scaled_test_data = test.drop(["label"],axis=1)
    return scaled_train_data, train_labels, scaled_test_data, test_labels
    
