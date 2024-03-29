# Logistic Regression with Gradient Descent

<H4>Initialization:</H4>
Create an instance of the LogisticReg class by providing an optional parameter focus which defaults to 10. For example:


```logistic_model = LogisticReg(focus=15)```

<H4>Training the model:</H4>
Use the fit_model method to train the logistic regression model by passing training data X_tr (input features) and y_tr (target variable). For instance:


```logistic_model.fit_model(X_train, y_train)```

<H4>Making predictions:</H4>
After training the model, you can make predictions on a test dataset (test_list) using the make_predictions method. For example:


```predictions = logistic_model.make_predictions(X_test)```

<H4>Evaluating accuracy:</H4>

Calculate the accuracy of the model's predictions using the calculate_accuracy method by passing the predicted values (predictions) and the true values (y_test). For instance:


```accuracy = LogisticReg.calculate_accuracy(predictions, y_test)```

