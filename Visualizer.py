from BE import KNN
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import plotly.express as px
import plotly.graph_objects as go

class visualizer():
    
    def askData():
        print("Type the sepal width:")
        x = int(input())
        print("Type the sepal length")
        y = int(input())

        #The classes matrix works with [y, x] so i have to change the order of my array
        classesData = [y, x]
        
        #The position works as usual, so i need it in the normal order
        positionData = [x, y]
        data = [classesData, positionData]
        
        return data


    def __init__(self):
        iris = load_iris()
        self.Xvar = iris.data
        self.Yvar = iris.target

        X_train, X_test, y_train, y_test = train_test_split(visualizer.Xvar, visualizer.Yvar, test_size=0.3)

        data, classes = load_iris(return_X_y=True)
        
    def start():      
        k = 3
        KNN.__init__(KNN, k)
        KNN.fit(KNN, visualizer.Xvar, visualizer.Yvar)
 
        
        data, classes = load_iris(return_X_y=True)
        
        #Graphic    
        df = px.data.iris()
        df = df[["sepal_width", "sepal_length", "species"]]
        fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
        fig.show()
        return df
            

    def addDot(df, classData, positionData):

        
        #Para los cercanos
        vecinos = KNN.get_k_nearest_neighboors(KNN, classData)
        vecinos = np.array(vecinos)
        Y = KNN.predict(KNN, visualizer.Xvar)
        
        preds = []
        
        #Classes numeration
        classes = [0, 1, 2]
        
        #Classes count
        count = [0, 0, 0]
        count = np.array(count)
        for i in range(len(vecinos)):
            preds.append(Y[vecinos[i]])
        
        for i in range(len(preds)):
            for j in range(len(classes)):
                if(preds[i] == classes[j]):
                    count[j] += 1
                     
        mostRepitedValue = 0
        
        for i in range(len(count)-1):
            if(count[i] > count[i+1]):
                mostRepitedValue = i

        detectedSpecie = preds[mostRepitedValue]
        
        if(detectedSpecie == 0):
            detectedSpecie = "setosa"
        elif(detectedSpecie == 1):
            detectedSpecie = "versicolor"
        elif(detectedSpecie == 2):
            detectedSpecie = "virginica"
        else:
            print("error")
    
        newDyc = {"sepal_width": positionData[0], "sepal_length": positionData[1], "species": detectedSpecie}
        df = df.append(newDyc, ignore_index=True)
        fig2 = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
        fig2.show()
        
if __name__ == '__main__':
    visualizer.__init__(visualizer)
    df = visualizer.start()
    data = visualizer.askData()
    visualizer.addDot(df, data[0], data[1])
    

    