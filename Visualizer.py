from plotly.graph_objs import layout
from BE import KNN
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import plotly.express as px
import plotly.graph_objects as go


class visualizer():

    def choose_dataset():
        print("\n")

        choosed = input("Type the name of the csv: ")
        data_set = pd.read_csv("Datasets/"+choosed+".csv")

        independent_variable1 = int(input(
            "Type the column position of the dependent variable 1 (x) (must be numeric) from left to right: "))-1
        independent_variable2 = int(input(
            "Type the column position of the dependent variable 2 (y) (must be numeric) from left to right: "))-1

        dependent_variable = int(input(
            "Type the column position of the independent variable (must be numeric) from left to right: "))-1

        # Verificar que el archivo sea .csv

        # Forma óptima
        temp = data_set.values
        x = data_set.iloc[:, [independent_variable1, independent_variable2]]

        # La dependiente no se normaliza
        # Normalización ( hacer para las independientes):
        x = (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))

        y = data_set.iloc[:, dependent_variable]

        KNN.fit(KNN, x, y)

        # Tomar etiquetas posibles
        numColumn = y.factorize()[1]
        tagColumn = y.factorize()[0]

        # Para tomar el nombre poner el índice en el tag column

        vid1 = data_set.columns[independent_variable1] + ""
        vid2 = data_set.columns[independent_variable2] + ""
        dv1 = data_set.columns[dependent_variable] + ""

        df = data_set[[vid1, vid2, dv1]]
        fig = px.scatter(df, x=vid1, y=vid2, color=dv1)
        fig.show()
        return df

    def fit_v(x, y):
       data, classes = x, y

    def askData():
        dfR = visualizer.choose_dataset()

        k = int(input("How many neighbors: "))
        knn = KNN(k)

        x1 = float(input("Type the value of  feature 1 (x): "))
        x2 = float(input("Type the value of feature 2 (y): "))

        # The classes matrix works with [y, x] so i have to change the order of my array
        classesData = [x1, x2]

        # The position works as usual, so i need it in the normal order
        positionData = [x1, x2]
        data = [classesData, positionData]

        visualizer.addDot(dfR, classesData, positionData)
        return data

    def __init__(self):
        self.start()

    def start():
        data = visualizer.askData()
        # Graphic

        # Df = el dataset que carguen
        return df

    def addDot(df, classData, positionData):

        # Para los cercanos
        vecinos = KNN.get_k_nearest_neighboors(KNN, classData)
        vecinos = np.array(vecinos)
        Y = KNN.predict(KNN, visualizer.Xvar)

        preds = []

        # Classes numeration
        classes = [0, 1, 2]

        # Classes count
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

        # Hacer escalable

       # Hacer que lo detecte desde  la selección de columnas para tomar los nombres como es

        """"
        if(detectedSpecie == 0):
            detectedSpecie = "setosa"
        elif(detectedSpecie == 1):
            detectedSpecie = "versicolor"
        elif(detectedSpecie == 2):
            detectedSpecie = "virginica"
        else:
            print("error")
       
        Se crea el diccionario y luego se añade al df que ya teníamos
              
        newDyc = {"sepal_width": positionData[0], "sepal_length": positionData[1], "species": detectedSpecie}
        df = df.append(newDyc, ignore_index=True)
        fig2 = px.scatter(df, x="sepal_width", y="sepal_length", color="species")  
        visualizer.connect_dots(vecinos, classData, visualizer.Xvar, fig2)
        """

  
    def connect_dots(neighbours_position, new_dot, main_matrix, fig2):        
        dot_x = new_dot[1]
        dot_y = new_dot[0] 
        
        
        fig2.show()
        # Tenemos la posición, hay que tomarlo del arreglo con todos los datos para poder conectarlos
        
        for i in range(len(neighbours_position)):
            neighbour_dot_position = neighbours_position[i-1]
            neighbour_dot  = visualizer.Xvar[neighbour_dot_position]
            old_dot_class = visualizer.Yvar[neighbour_dot_position] 
            
            neighbour_x = neighbour_dot[1]
            neighbour_y = neighbour_dot[0]
            
            fig2.add_scatter(x=[neighbour_x, dot_x], y=[neighbour_y, dot_y], showlegend=False)
        

        fig2.show()

        
if __name__ == '__main__':
    visualizer.__init__(visualizer)
    df = visualizer.start()
    # data = visualizer.askData()
    # visualizer.addDot(df, data[0], data[1])
    
    

    
