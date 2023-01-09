# write main starter code

# import 
from utils import UNSW, NSL_KDD, AWID, get_test_accuracy_of
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from model import Model


def main():
    for dataset in [UNSW, NSL_KDD, AWID]:
        x_train, y_train, x_test, y_test = dataset()
        model = Model(x_train.shape[1], "100-100-50", x_train, y_train, x_test, y_test)
        model.train(10)
        train_preds, train_labels, test_preds, test_labels = model.compute_vectors()
        for model in [RandomForestClassifier(), DecisionTreeClassifier(), LogisticRegression(), SVC(), MLPClassifier()]:
            model.fit(train_preds, train_labels)
            print(get_test_accuracy_of(model, test_preds, test_labels))


if __name__ == "__main__":
    main()
            