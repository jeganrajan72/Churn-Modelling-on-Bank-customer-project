import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv(r"C:\Users\asus\Downloads\bank churn modell\train.csv")


# Split data into features and target
X = df.drop(['Exited'], axis=1)
y = df['Exited']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101,stratify=y)


# Function to drop string variables
def drop_string_variables(X):
    return X.drop(['id', 'CustomerId', 'Surname'],axis=1)



pipe = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('drop_string', FunctionTransformer(drop_string_variables), X.columns.drop(['Geography', 'Gender'])),  # Apply to all non-categorical columns
        ('onehot', OneHotEncoder(), ['Geography', 'Gender'])  # Apply one-hot encoding to specified columns
    ], remainder='passthrough')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

pipe.fit(X_train, y_train)
print(pipe.score(X_train, y_train))



import joblib
joblib.dump(pipe, "saved_the_pipe_model.pkl")
print("The model is saved")