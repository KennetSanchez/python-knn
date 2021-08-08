import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import plotly.express as px
import plotly.graph_objects as go

def askData():
    print("Type the sepal width:")
    x = int(input())
    print("Type the sepal length")
    y = int(input())

    newData = [x, y]
    return newData


def start():
    iris = load_iris()
    X = iris.data
    Y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    
    """
    print(type(X_train))
    print(X_train)
    print(type(y_train))
    print(y_train)
    """
    data, classes = load_iris(return_X_y=True)
    
    #Graphic    
    df = px.data.iris()
    df = df[["sepal_width", "sepal_length", "species"]]
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
    fig.show()
    return df
          

def addDot(df):
    newData = askData()

    newDyc = {"sepal_width": newData[0], "sepal_length": newData[1], "species": "setosa"}
    df = df.append(newDyc, ignore_index=True)
    fig2 = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
    fig2.show()
    
if __name__ == '__main__':
    df = start()
    addDot(df)