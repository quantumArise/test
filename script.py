import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn import metrics
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['Unnamed: 0'])
    df = df.dropna()
    df['class'] = df['class'].replace({'A': 0, 'B': 1})
    y = df.pop("class").values
    X = df.values
    return X, y

def preprocess_data(X):
    PredictorScaler = StandardScaler()
    PredictorScalerFit = PredictorScaler.fit(X)
    X = PredictorScalerFit.transform(X)
    return X

def train_models(X_train, y_train):
    rf = RandomForestClassifier()
    baseline_model = rf.fit(X_train, y_train)
    
    ros = RandomOverSampler(random_state=0)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
    ros_model = rf.fit(X_train_ros, y_train_ros)
    
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    
    clf_ros = SVC(kernel='linear', probability=True)
    clf_ros.fit(X_train_ros, y_train_ros)
    
    return baseline_model, ros_model, clf, clf_ros

def plot_roc(ax, X_train, y_train, X_test, y_test, title):
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict_proba(X_test)[:,1]
    fpr, tpr, thresh = metrics.roc_curve(y_test, y_test_pred)
    auc = metrics.roc_auc_score(y_test, y_test_pred)
    ax.plot(fpr, tpr, label=f"{title} AUC={auc:.2f}")
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate', fontsize="14")
    ax.set_ylabel('True Positive Rate', fontsize="14")
    ax.legend(loc=0, fontsize="14")
    ax.tick_params(axis='both', which='major', labelsize=14)

def main(file_path):
    X, y = load_data(file_path)
    X = preprocess_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print("Number of samples in the training set:", X_train.shape[0])
    print("Number of samples in the test set:", X_test.shape[0])
    print("Distribution of classes in the training set:", sorted(Counter(y_train).items()))
    
    baseline_model, ros_model, clf, clf_ros = train_models(X_train, y_train)
    
    baseline_prediction = baseline_model.predict(X_test)
    ros_prediction = ros_model.predict(X_test)
    clf_prediction = clf.predict(X_test)
    clf_ros_prediction = clf_ros.predict(X_test)
    
    print("Classification report for the baseline model:\n", classification_report(y_test, baseline_prediction))
    print("Classification report for the ROS model:\n", classification_report(y_test, ros_prediction))
    print("Classification report for the SVC model:\n", classification_report(y_test, clf_prediction))
    print("Classification report for the ROS SVC model:\n", classification_report(y_test, clf_ros_prediction))
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plot_roc(ax, X_train, y_train, X_test, y_test, 'Original Dataset')
    plot_roc(ax, X_train_ros, y_train_ros, X_test, y_test, 'Randomly Oversampled Dataset')
    plt.legend(loc='lower right', fontsize="14")
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 10
    plt.savefig("classification.pdf", dpi=1000, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py file_path")
        sys.exit(1)
    file_path = sys.argv[1]
    main(file_path)
