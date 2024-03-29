import numpy as np
import pandas as pd
from random import randint
from math import e, log


class LogisticReg:
    def __init__(self, focus=10):
        # Initialize class variables
        self.col_index = None  # Index of the column being processed
        self.len_columns = None  # Number of columns in the dataset
        self.X = None  # Input features
        self.y = None  # Target variable
        self.loop_check = False  # Flag indicating whether the random step generation loop has been repeated 5 times
        self.done_situation = False  # Flag indicating if convergence check is complete
        self.minimum_gradient = []  # List to store minimum gradients
        self.focus = focus  # Number of focus points for optimization

    def fit_model(self, X_tr, y_tr):
        # Fit the logistic regression model
        self.X, self.y = X_tr, y_tr  # Set the training data
        self.len_columns = len(self.X.columns)  # Calculate the number of columns
        for col in range(self.len_columns):
            print("yo")
            self.col_index, self.done_situation, self.loop_check = col, False, False
            # Initialize column index and convergence flag
            start_step = self.find_initial_step()  # Find the initial step for optimization
            self.update_minimum_gradient(start_step)  # Update minimum gradient

    def find_initial_step(self):
        # Find the initial step for optimization
        while True:
            i = randint(-100, 100)  # Generate a random integer
            self.loop_check = False  # Reset loop flag
            result = self.calculate_function_minimum(i, first_step_check=True)  # Calculate function minimum
            if result != 0:
                break  # Exit loop if result is not zero
        return result  # Return the initial step

    def calculate_function_minimum(self, i, position="", first_step_check=False, loop_check=0):
        # Calculate the function minimum for a given step
        if loop_check == 5 or self.loop_check:
            self.loop_check = True
            return  # Exit if loop limit is reached
        status, total, counter = [], 0, 0  # Initialize status, total, counter

        for j in self.X.iloc:
            if np.isnan(j[self.col_index]):
                raise NameError("Nan value")
            point = j[self.col_index] * i  # Calculate the point
            point = 1 / (1 + e ** -point)  # Apply sigmoid function
            if (self.y[j.name] == 1 and point == 0) or (self.y[j.name] == 0 and point == 1):
                counter += 1  # Increment counter
                break  # Exit loop
            if self.y[j.name] == 1:
                total += log(point)  # Update total
            elif self.y[j.name] == 0:
                total += log(1 - point)  # Update total

        if counter == 0:
            status = (i, total)  # Set status if counter is zero
        elif counter != 0 and not first_step_check:
            if position == "previous_step":
                self.calculate_function_minimum(i - 1, position="previous_step", loop_check=loop_check + 1)
            elif position == "next_step":
                self.calculate_function_minimum(i + 1, position="next_step", loop_check=loop_check + 1)

        if len(status) > 0:
            return status  # Return status if valid
        else:
            return 0  # Return zero if invalid

    def update_minimum_gradient(self, start):
        # Update the minimum gradient recursively
        previous_step = self.calculate_function_minimum(start[0] - 1, position="previous_step")
        self.loop_check = False  # Reset loop_check flag
        next_step = self.calculate_function_minimum(start[0] + 1, position="next_step")
        self.loop_check = False  # Reset loop_check flag
        self.check_convergence(previous_step, start, next_step)

    def check_convergence(self, pre, main, nxt):
        # Check for convergence during optimization
        self.done_situation = False  # Reset convergence flag

        if type(pre) == int and type(nxt) == int:
            self.set_domain(main)  # Set domain for updating pixel values
            return

        if type(pre) == int or type(nxt) == int:
            if type(pre) == int:
                if nxt[1] <= main[1]:
                    self.set_domain(main)  # Set domain for updating pixel values
                    return
                else:
                    self.update_minimum_gradient(nxt)
            else:
                if pre[1] <= main[1]:
                    self.set_domain(main)  # Set domain for updating pixel values
                    return
                else:
                    self.update_minimum_gradient(pre)
        if self.done_situation:
            return

        if main[1] >= nxt[1] and main[1] >= pre[1]:
            self.set_domain(main)  # Set domain for updating pixel values
            return

        if pre[1] >= main[1] and pre[1] >= nxt[1]:
            self.update_minimum_gradient(pre)
        else:
            self.update_minimum_gradient(nxt)

    def set_domain(self, main_num):  # Set domain for updating pixel values
        domain = [main_num[0] - 1, main_num[0] + 1]  # Define domain for updating pixel values
        self.done_situation = True  # Set convergence flag
        self.update_pixel_values(domain)  # Update pixel values

    def update_pixel_values(self, domain):
        # Update pixel values during optimization
        slice_values = np.linspace(domain[0], domain[1], self.focus)  # Generate values within the domain
        status = [self.calculate_function_minimum(i) for i in slice_values]  # Calculate function minimum
        status = pd.DataFrame([i for i in status if i != 0])  # Filter non-zero values
        index = status.iloc[status[1].idxmax()]  # Get index of maximum value
        self.minimum_gradient.append(index[0])  # Append maximum value to minimum gradient list

    def make_predictions(self, test_list):
        # Make predictions using the trained model
        predict_list = []
        for i in test_list.iloc:
            point = 0  # Initialize point
            for j in range(self.len_columns):
                point += i[j] * self.minimum_gradient[j]  # Calculate weighted sum
            point = 1 / (1 + e ** -point)  # Apply sigmoid function
            if point >= 0.5:
                predict_list.append(1)  # Append 1 for positive prediction
            else:
                predict_list.append(0)  # Append 0 for negative prediction

        return predict_list  # Return the list of predictions

    @staticmethod
    def calculate_accuracy(pred_list, true_list):
        # Calculate accuracy based on predictions and true values
        data_zipped = zip(pred_list, true_list.values)  # Zip predictions and true values
        n_True = [1 for sample in data_zipped if sample[0] == sample[1]]  # Count true predictions
        return sum(n_True) / len(true_list)  # Calculate accuracy
