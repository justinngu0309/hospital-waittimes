# Import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score, log_loss, classification_report, recall_score
from sklearn.utils.class_weight import compute_class_weight

# Load and prepare data
full_df = pd.read_csv('Hospitalwaittimecharge.csv')

# Transform target into binary
full_df['LengthOfStay'] = full_df['LengthOfStay'].apply(lambda x: 0 if x <= 3 else 1).astype('int')

# Split into train (80%) and test (20%)
train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42, 
                                    stratify=full_df['LengthOfStay'])

# Data cleaning
train_df = train_df.loc[train_df['Gender']!='U']

# Convert categorical variables
train_df['CCSProcedureCode'] = train_df['CCSProcedureCode'].astype('object')
test_df['CCSProcedureCode'] = test_df['CCSProcedureCode'].astype('object')

# Normalize severity code
train_df['APRSeverityOfIllnessCode'] = train_df['APRSeverityOfIllnessCode'].astype(np.int64)/4.0
test_df['APRSeverityOfIllnessCode'] = test_df['APRSeverityOfIllnessCode'].astype(np.int64)/4.0

# One-hot encoding function
def oneHotEncodeColumn(df, colName, drop_col, enc_onehot=None):
    if enc_onehot is None:
        enc_onehot = OneHotEncoder(handle_unknown='ignore')
        enc_onehot.fit(df[[colName]])

    onehot_ = enc_onehot.transform(df[[colName]]).toarray()

    for i in range(len(enc_onehot.categories_[0])):
        if enc_onehot.categories_[0][i] != drop_col:
            col_name = f"{colName}_{enc_onehot.categories_[0][i]}"
            df[col_name] = onehot_[:,i]

    df = df.drop([colName], axis=1)
    return df, enc_onehot

# Apply one-hot encoding
train_df, gender_enc = oneHotEncodeColumn(train_df, 'Gender', 'U')
train_df, race_enc = oneHotEncodeColumn(train_df, 'Race', 'Other Race')
train_df, ToA_enc = oneHotEncodeColumn(train_df, 'TypeOfAdmission', 'Newborn')
train_df, CCSPC_enc = oneHotEncodeColumn(train_df, 'CCSProcedureCode', -1)
train_df, paymentTop_enc = oneHotEncodeColumn(train_df, 'PaymentTypology', 'Miscellaneous/Other')
train_df, Egcy_enc = oneHotEncodeColumn(train_df, 'EmergencyDepartmentIndicator', 'N')

# Split train into train and validation
data_trainSplit = train_df[(train_df['HealthServiceArea'] == 'New York City') | 
                         (train_df['HealthServiceArea'] == 'Southern Tier')]
data_valSplit = train_df[(train_df['HealthServiceArea'] == 'Long Island') | 
                       (train_df['HealthServiceArea'] == 'Central NY')]

# Scale numerical features
cols2scale = ['AverageCostInCounty', 'AverageChargesInCounty',
             'AverageCostInFacility', 'AverageChargesInFacility',
             'AverageIncomeInZipCode', 'BirthWeight']

minmax_scaler_hold = {}
for col in cols2scale:
    scaler = MinMaxScaler()
    data_trainSplit[col] = scaler.fit_transform(data_trainSplit[[col]])
    data_valSplit[col] = scaler.transform(data_valSplit[[col]])
    test_df[col] = scaler.transform(test_df[[col]])
    minmax_scaler_hold[col] = scaler

# Prepare features and targets
X_train = data_trainSplit.drop(['LengthOfStay', 'HealthServiceArea', 'ID'], axis=1)
y_train = data_trainSplit['LengthOfStay'].astype(np.int64)
X_val = data_valSplit.drop(['LengthOfStay', 'HealthServiceArea', 'ID'], axis=1)
y_val = data_valSplit['LengthOfStay'].astype(np.int64)
X_test = test_df.drop(['LengthOfStay', 'HealthServiceArea', 'ID'], axis=1)
y_test = test_df['LengthOfStay'].astype(np.int64)

# Class weights for imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(zip(np.unique(y_train), class_weights))

# Model training and evaluation function
def fit_classification_model(clf, xtrain, ytrain, xval, yval, class_weights, print_report=False):
    sample_weights = np.where(ytrain==1, class_weights[1], class_weights[0])
    clf.fit(xtrain, ytrain, sample_weight=sample_weights)
    
    ytrain_pred = clf.predict(xtrain)
    yval_pred = clf.predict(xval)
    
    f1_train = f1_score(ytrain, ytrain_pred, average='macro')
    f1_val = f1_score(yval, yval_pred, average='macro')
    
    if print_report:
        print(classification_report(yval, yval_pred))
        
    return clf, f1_train, f1_val

# Train and compare models
results = {}

# 1. Logistic Regression
lr = LogisticRegression(class_weight='balanced', max_iter=500)
lr, lr_train_f1, lr_val_f1 = fit_classification_model(lr, X_train, y_train, X_val, y_val, class_weights)
results['Logistic Regression'] = {'model': lr, 'train_f1': lr_train_f1, 'val_f1': lr_val_f1}

# 2. Decision Tree
dt = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=42)
dt, dt_train_f1, dt_val_f1 = fit_classification_model(dt, X_train, y_train, X_val, y_val, class_weights)
results['Decision Tree'] = {'model': dt, 'train_f1': dt_train_f1, 'val_f1': dt_val_f1}

# 3. Random Forest
rf = RandomForestClassifier(n_estimators=500, min_samples_split=2, 
                          min_samples_leaf=2, max_depth=15,
                          class_weight='balanced', random_state=42)
rf, rf_train_f1, rf_val_f1 = fit_classification_model(rf, X_train, y_train, X_val, y_val, class_weights)
results['Random Forest'] = {'model': rf, 'train_f1': rf_train_f1, 'val_f1': rf_val_f1}

# 4. AdaBoost
ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.00025, random_state=42)
ada, ada_train_f1, ada_val_f1 = fit_classification_model(ada, X_train, y_train, X_val, y_val, class_weights)
results['AdaBoost'] = {'model': ada, 'train_f1': ada_train_f1, 'val_f1': ada_val_f1}

# Print results
print("\nModel Performance Comparison:")
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  Train F1: {metrics['train_f1']:.4f}")
    print(f"  Val F1: {metrics['val_f1']:.4f}\n")

# Make final predictions on test set with best model
best_model_name = max(results.items(), key=lambda x: x[1]['val_f1'])[0]
best_model = results[best_model_name]['model']
test_pred = best_model.predict(X_test)

print(f"\nBest Model: {best_model_name}")
print("Test Set Classification Report:")
print(classification_report(y_test, test_pred))

# Save predictions
test_df['Predicted_LengthOfStay'] = test_pred
test_df[['ID', 'Predicted_LengthOfStay']].to_csv('predictions.csv', index=False)