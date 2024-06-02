import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, precision_score, \
    recall_score
import joblib
from matplotlib import pyplot as plt
import seaborn as sns


# Function to train and evaluate the model, and plot confusion matrix
def train_and_evaluate(df, model_filename, confusion_matrix_filename):
    # Encode the 'HandType' column
    df_encoded = pd.get_dummies(df, columns=['HandType'], drop_first=True)

    # Separate features (X) and target variable (y)
    X = df_encoded.drop(['Class', 'ImageName'], axis=1)  # Use all columns except 'Class' and 'ImageName'
    y = df['Class']

    # Ensure the target variable y is of a valid type for classification
    if not pd.api.types.is_categorical_dtype(y):
        y = pd.Categorical(y).codes

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Evaluate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    prec_score = precision_score(y_test, y_pred, average='weighted')
    r_score = recall_score(y_test, y_pred, average='weighted')
    fscore = f1_score(y_test, y_pred, average='weighted')

    print(f"Precision score of Random Forest: {prec_score:.5f}")
    print(f"Recall score of Random Forest: {r_score:.5f}")
    print(f"F1 score of Random Forest: {fscore:.5f}")
    print(f"Confusion Matrix of Random Forest:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report of Random Forest:\n{classification_report(y_test, y_pred, zero_division=0)}")

    # Save the trained model to a file
    joblib.dump(model, model_filename)

    # Plot confusion matrix
    classes = np.unique(y_test)
    fig, ax = plt.subplots(figsize=(12, 8))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
    ax.yaxis.set_major_locator(plt.FixedLocator(range(len(classes))))

    ax.set_yticklabels(labels=classes, rotation=0)
    plt.savefig(confusion_matrix_filename)
    plt.close(fig)


# Load datasets for Alphabets and Numbers
df_alphabets = pd.read_csv('ASL_ALPHABETS.csv')
df_numbers = pd.read_csv('ASL_NUMBERS.csv')

# Ensure 'Class' column is present in both datasets
if 'Class' not in df_alphabets.columns:
    raise KeyError("'Class' column is missing from the alphabets dataset")
if 'Class' not in df_numbers.columns:
    raise KeyError("'Class' column is missing from the numbers dataset")

# Train and evaluate the model for alphabets
train_and_evaluate(df_alphabets, 'ASL_ALPHABETS_MODEL.joblib', 'Confusion_RF_Alphabets.png')

# Train and evaluate the model for numbers
train_and_evaluate(df_numbers, 'ASL_NUMBERS_MODEL.joblib', 'Confusion_RF_Numbers.png')
