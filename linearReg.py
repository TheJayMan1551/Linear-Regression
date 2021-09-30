import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm

# reading data from the csv
data = pd.read_csv('grades.csv')
dataNum = pd.read_csv('gradesNum.csv')

# sns.lmplot(x='gpa', y='rating', data=data, hue='Courses', fit_reg=False)
# plt.show()

x = dataNum['x']
y = dataNum['y']
print(data.head())

print("")


def linear_regression(x, y):
    N = len(x)
    x_mean = x.mean()
    y_mean = y.mean()

    B1_num = ((x - x_mean) * (y - y_mean)).sum()
    B1_den = ((x - x_mean)**2).sum()
    B1 = B1_num / B1_den

    B0 = y_mean - (B1*x_mean)

    reg_line = 'y = {} + {}β'.format(B0, round(B1, 3))

    return (B0, B1, reg_line)


def corr_coef(x, y):
    N = len(x)

    num = (N * (x*y).sum()) - (x.sum() * y.sum())
    den = np.sqrt((N * (x**2).sum() - x.sum()**2)
                  * (N * (y**2).sum() - y.sum()**2))
    R = num / den
    return R


B0, B1, reg_line = linear_regression(x, y)
print('Regression Line: ', reg_line)
R = corr_coef(x, y)
print('Correlation Coef.: ', R)
print('"Goodness of Fit": ', R**2)

x = dataNum['x'].tolist()
y = dataNum['y'].tolist()
plt.scatter(x, y)

x = sm.add_constant(x)
result = sm.OLS(y, x).fit()
print(result.summary())

max_x = 10
min_x = 0

x = np.arange(min_x, max_x, 1)
y = 0.1053 * x + 2.6795
plt.plot(y, 'r')
plt.show()
