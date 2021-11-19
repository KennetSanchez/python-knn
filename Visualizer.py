from plotly.graph_objs import layout
from BE import KNN
import numpy as np
#from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
#from sklearn.datasets import load_iris
import plotly.express as px
import plotly.graph_objects as go


class Visualizer():

    def choose_dataset(self):
        print("\n")

        choosed = input("Type the name of the csv: ")
        self.data_set = pd.read_csv("Datasets/"+choosed+".csv")

        self.independent_variable1 = int(input(
            "Type the column position of the independent variable 1 (x) (must be numeric) from left to right: "))-1
        self.independent_variable2 = int(input(
            "Type the column position of the independent variable 2 (y) (must be numeric) from left to right: "))-1

        dependent_variable = int(input(
            "Type the column position of the dependent variable (must be numeric) from left to right: "))-1

        # Verificar que el archivo sea .csv

        # Forma óptima

        x = self.data_set.iloc[:, [self.independent_variable1, self.independent_variable2]]
        x = x.astype("float64")
        x = x.values
        self.X_whitout_normalization = x
        # La dependiente no se normaliza
        # Normalización ( hacer para las independientes):

        self.xmin = x.min(axis=0)
        self.xmax = x.max(axis=0)

        self.Xvar = (x-self.xmin)/(self.xmax-self.xmin)

        y = self.data_set.iloc[:, dependent_variable]
        self.Yvar = y.factorize()[0]

        self.categories = y.unique()

        self.knn.fit(self.X_whitout_normalization, self.Yvar)

        # Tomar etiquetas posibles
        # self.tagColumn = y.factorize()[1]
        # self.numColumn = y.factorize()[0]

        # Para tomar el nombre poner el índice en el tag column

        self.vid1 = self.data_set.columns[self.independent_variable1] + ""
        self.vid2 = self.data_set.columns[self.independent_variable2] + ""
        self.dv1 = self.data_set.columns[dependent_variable] + ""


        df = self.data_set[[self.vid1, self.vid2, self.dv1]]

        fig = px.scatter(df, x=self.vid1, y=self.vid2, color= self.dv1)


        fig.show()
        return df

    def fit_v(x, y):
       data, classes = x, y

    def askData(self):
        self.dfR = self.choose_dataset()
        k = int(input("How many neighbors do you want? "))
        self.knn.k = k

        self.position_x = float(input("Type the value of  feature 1 (x): "))
        self.position_y = float(input("Type the value of feature 2 (y): "))

        # The classes matrix works with [y, x] so i have to change the order of my array
        x1 = self.position_x#= self.position_x-self.xmin/(self.xmax-self.xmin)
        x2 = self.position_y#= self.position_y-self.xmin/(self.xmax-self.xmin)

        classesData = [x1, x2]

        self.addDot(classesData)
        return classesData

    def __init__(self, knn):
        self.knn = knn
        self.Xvar = None
        self.Yvar = None
        self.dfR = None
        self.tagColumn = None
        self.numColumn = None
        self.vid1 = None
        self.vid2 = None
        self.dv1 = None
        self.xmin = None
        self.xmax = None
        self.categories = None
        self.X_whitout_normalization = None
        self.independent_variable1  = None
        self.independent_variable2  = None
        self.position_x = None
        self.position_y = None
        self.data_set = None




    def start(self):
        data = self.askData()
        return data

    ##Borrar positionData
    def addDot(self, classData):

        # Para los cercanos
        vecinos = self.knn.get_k_nearest_neighboors(classData)
        vecinos = np.array(vecinos)
        Y = self.knn.predict(self.Xvar)

        preds = []
        count = []

        # Classes numeration
        classes = []
        for i in range (len(self.categories)):
            classes.append(i-1)
            count.append(0)

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

        detectedClass = preds[mostRepitedValue]
        detectedClass = self.categories[detectedClass]

        newDyc = {self.vid1: self.position_x, self.vid2: self.position_y, self.dv1: detectedClass}

        self.data_set.append(newDyc, ignore_index = True)

        fig2 = px.scatter(self.dfR, x=self.vid1, y=self.vid2, color=self.dv1)
        #fig2.show()

        self.connect_dots(vecinos, classData, fig2)


    def connect_dots(self, neighbours_position, position_data, fig2):
  
        print(neighbours_position)

        for i in range(len(neighbours_position)):
            neighbour_dot_position = neighbours_position[i]
            neighbour_dot  = self.X_whitout_normalization[neighbour_dot_position]

            neighbour_y = neighbour_dot[1]
            neighbour_x = neighbour_dot[0]

            fig2.add_scatter(x=[neighbour_x, self.position_x], y=[neighbour_y, self.position_y], showlegend=False, line_color= "green")


        fig2.show()


if __name__ == '__main__':
    knn = KNN()
    vis = Visualizer(knn)
    data = vis.start()
