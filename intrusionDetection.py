import csv
import pandas
import sys
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

from PIL import Image


def main():
    training("GaussianNB", "kdd_data.csv", "kdd_test.csv")


def training(selected_model , train_file_path, test_file_path):
    cv = 3
    if(selected_model is None):
        raise Exception("Data usage model")
        return -1
    if(train_file_path is None):
        raise Exception("Data usage train file")
        return -1
    if(test_file_path is None):
        raise Exception("Data usage test file")
        return -1
    # Load data from spreadsheet and split into train and test sets
    train_data, labels_train = load_data(train_file_path)
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(train_data)
    test_data, labels_test = load_data(test_file_path)
 
    # Train model and make predictions
    model = train_model(train_data, labels_train,selected_model, cv)
    # cv_scores = cross_val_score(model, train_data, labels_train, cv=StratifiedKFold(
    #     n_splits=cv), scoring='accuracy')
    # print(f"Cross-validation scores: {cv_scores}")
    # print(f"Mean Accuracy: {cv_scores.mean()}")
    model.fit(train_data, labels_train)
    predictions = model.predict(test_data)
    sensitivity, specificity = evaluate(labels_test, predictions)

    # Print results
    print(f"Correct: {(labels_test == predictions).sum()}")
    print(f"Incorrect: {(labels_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")
    
    return sensitivity , specificity, labels_test, predictions


def load_data(filename):

    evidence = []
    labels = []
    print("this is read data step")
    # read filename
    csv_file = pandas.read_csv(filename)

    labels_df = []

    # different file name has different features
    
    label_encoder = LabelEncoder()  
    print("====DATASET 1=====")
    csv_file['label'] = csv_file['label'].apply(lambda x: 0 if x == 'normal.' else 1)
    labels_df = csv_file['label']
    evidence_df = csv_file.drop(columns=['label'])
    features = ['protocol_type' ,'service', 'flag', 'is_host_login']
    
    for feature in features:
        evidence_df[feature] = label_encoder.fit_transform(evidence_df[feature])
    print(evidence_df.shape)
    print(labels_df.shape)
    evidence_list = evidence_df.values.tolist()
    labels_list = labels_df.values.tolist()
    #img = px.histogram(csv_file , x="label" ,  color="label", title="Images By label ")
    #img.show()
    #img.write_image(f"label_img.png")
    return evidence_list, labels_list



def train_model(evidence, labels, selected_model, cv):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    print("this is train model step")

    print(selected_model)
    if selected_model == "LogisticRegression":
        model = LogisticRegression(solver="saga")
    elif selected_model == "RandomForestClassifier":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif selected_model == "SVC":
        model = SVC(kernel='linear', C=1.0)
    elif selected_model == "MLPClassifier":
        model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 50), random_state=1)
    elif selected_model == "GaussianNB":
        model = GaussianNB()
    else:
        model = DecisionTreeClassifier()


    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    print("this is evaluate step")
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel() 
    TPR = tp / (tp + fn)
    TNR = tn / (tn + fp)
    # Bar chart
    labels = ['Correct', 'Incorrect']
    counts = [tp, tn]
    colors = ['green', 'red']

    plt.bar(labels, counts, color=colors)
    plt.xlabel('Classification')
    plt.ylabel('Count')
    plt.title('Correct and Incorrect Classifications')

    # Displaying true positive and true negative rates as text on the plot
    plt.text(0, tp + 1000, f'True Positive Rate: {100*TPR:.2f}%', ha='center', va='center', color='blue')
    plt.text(1, tn + 1000, f'False Negative Rate: {100*(1-TNR):.2f}%', ha='center', va='center', color='blue')
    plt.savefig('correct_img.png')

    plt.show()

    return TPR, TNR

if __name__ == "__main__":
    main()
