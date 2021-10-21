from sklearn import tree, model_selection
import pandas as pd

gradesData = pd.read_csv('grades.csv')

del gradesData['Courses']

x = gradesData.drop('gpa', axis=1)
y = gradesData['gpa']

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.2, random_state=0)

rating = ['rating']

algo = tree.DecisionTreeRegressor()
algo.fit(X_train, y_train)

showData = tree.export_text(algo, feature_names=rating, max_depth=10)

predict_db = pd.DataFrame({"rating": 9}, index=[1])

prediction = algo.predict(predict_db)

print(f'Decision Tree\n=====================\n\n{showData}')

print(f'Machine Learning predicted grade: {prediction}')
