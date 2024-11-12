import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

from explainerdashboard import RegressionExplainer, ExplainerDashboard
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

if __name__ == '__main__':
    df = pd.read_csv('preprocessed_data.csv')

    X = df.drop(columns=['Rings', 'Sex'])
    y = df['Rings']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression().fit(X_train, y_train)

    explainer = RegressionExplainer(model, X_test, y_test)

    db = ExplainerDashboard(explainer)
    db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)
