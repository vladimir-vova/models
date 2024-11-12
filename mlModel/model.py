import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

if __name__ == '__main__':
    df = pd.read_csv('preprocessed_data.csv')

    X = df.drop(columns=['Rings', 'Sex'])
    y = df['Rings']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # X_test.to_csv('X_test.csv', index=False)
    # y_test.to_csv('y_test.csv', index=False)

    model = LinearRegression()

    model.fit(X_train, y_train)

    with open('trained_model.cbm', 'wb') as model_file:
        pickle.dump(model, model_file)