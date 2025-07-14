import pandas as pd
import numpy as np
import math

df = pd.read_csv(r'C:\Users\SESA610197\Documents\AIMLLNG\ML\Gradient_Descent and Cost_function\Exercise\test_scores.csv')
print(df)

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.0001

    cost_previous = 0
     
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * np.sum((y - y_predicted)**2)
        md = -(2/n) * np.sum(x * (y - y_predicted))
        bd = -(2/n) * np.sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
            
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost
        print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))

x = np.array(df.math)
y = np.array(df.cs)

gradient_descent(x,y)
