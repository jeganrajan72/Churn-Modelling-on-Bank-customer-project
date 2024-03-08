from flask import Flask,render_template,request
import pandas as pd
import joblib
from train import drop_string_variables


model = joblib.load("saved_the_pipe_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['post'])
def predict():
    id =int(request.form.get("id"))
    CustomerId =int(request.form.get("CustomerId"))
    Surname =request.form.get("Surname")
    CreditScore =int(request.form.get("CreditScore"))
    Geography =request.form.get("Geography")
    Gender =request.form.get("Gender")
    Age =int(request.form.get("Age"))
    Tenure =int(request.form.get("Tenure"))
    Balance =int(request.form.get("Balance"))
    NumOfProducts =int(request.form.get("NumOfProducts"))
    HasCreditCard =int(request.form.get("HasCreditCard"))
    IsActiveMember =int(request.form.get("IsActiveMember"))
    EstimatedSalary =int(request.form.get("EstimatedSalary"))

    dict = {'id': [id], 'CustomerId': [CustomerId], 'Surname': [Surname], 'CreditScore': [CreditScore],
            'Geography': [Geography], 'Gender': [Gender], 'Age': [Age], 'Tenure': [Tenure], 'Balance': [Balance],
            'NumOfProducts': [NumOfProducts], 'HasCrCard': [HasCreditCard], 'IsActiveMember': [IsActiveMember],
            'EstimatedSalary': [EstimatedSalary]}
    df = pd.DataFrame(dict)
    prediction= model.predict(df)
    prediction_result = "Predicted class: {}".format(prediction)
    # Here, you would pass these inputs to your model for prediction
    # For now, let's just print them
    print(f"ID: {id}")
    print(f"Customer ID: {CustomerId}")
    print(f"Surname: {Surname}")
    print(f"Credit Score: {CreditScore}")
    print(f"Geography: {Geography}")
    print(f"Gender: {Gender}")
    print(f"Age: {Age}")
    print(f"Tenure: {Tenure}")
    print(f"Balance: {Balance}")
    print(f"Number of Products: {NumOfProducts}")
    print(f"Has Credit Card: {HasCreditCard}")
    print(f"Is Active Member: {IsActiveMember}")
    print(f"Estimated Salary: {EstimatedSalary}")
    inputs = [id, CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts,
              HasCreditCard, IsActiveMember, EstimatedSalary]

    # Perform prediction
    print(prediction_result)

    # You can return the print values to the user as well

    return f"""
        <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{
                    background-color: #333; /* Dark background color */
                    color: white;
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                }}
                h1 {{
                    font-size: 24px;
                }}
                p {{
                    font-size: 16px;
                    margin-bottom: 5px;
                }}
                .prediction {{
                    font-size: 20px; /* Larger font for prediction */
                    margin-top: 20px;
                    background-color: #4CAF50; /* Optional color for highlighting */
                    padding: 10px;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <h1>Inputs Received</h1>
            <p>ID: {id}</p>
            <p>Customer ID: {CustomerId}</p>
            <p>Surname: {Surname}</p>
            <p>Credit Score: {CreditScore}</p>
            <p>Geography: {Geography}</p>
            <p>Gender: {Gender}</p>
            <p>Age: {Age}</p>
            <p>Tenure: {Tenure}</p>
            <p>Balance: {Balance}</p>
            <p>Number of Products: {NumOfProducts}</p>
            <p>Has Credit Card: {HasCreditCard}</p>
            <p>Is Active Member: {IsActiveMember}</p>
            <p>Estimated Salary: {EstimatedSalary}</p>
            <p class="prediction">Prediction Result: {prediction_result}</p>
        </body>
        </html>
        """
app.run(debug=True)