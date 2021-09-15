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
        
        dependent_variable = int(input("Type the column position of the dependent variable from left to right: "))
        
        ##Recibir 2 independientes para graficar; 1 dependiente para cambiar el color
        
        independent_variable = int(input("Type the column position of the independent variable from left to right: "))
        
        #Verificar que el archivo sea .csv
        
        ##Forma óptima
        temp = data_set.values
        f = data_set.iloc[:,1]
        f.min(axis = 0)
        
        
        #La dependiente no se normaliza
        #Normalización ( hacer para las independientes):
        temp = (temp-temp.min(axis = 0))/(temp.max(axis = 0)-temp.min(axis = 0))
        
        #Axis 0 filas; axis 1 columnas
              
        print(f)
        
        
        #data_set = data_set.columns
        
        #dv = data_set[dependent_variable]
        #iv = data_set[independent_variable]
        
        #Imprime es la categoría, no los valores
        #print(data_set['SepalLengthCm'])
        
        #Escoge por filas; no columnas
        #print(data_set.iloc[1, 'SepalLengthCm'])
        #data_set.iloc
        ##fit(dv, iv)
        
        
        ##Normalizar features antes de pasarlo al knn 
        # por cada variable (para cada índice) =  -mínimo / (máximo - mínimo)  queda entre 0 y 1

        
    def fit_v(x, y):
       data, classes = x, y     
        
        
        
    def askData():
        visualizer.choose_dataset()
        
        x = float(input("Type the sepal width: "))
        y = float(input("Type the sepal length: "))
       
        
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
        k = int(input("How many neighbors: "))
        KNN.__init__(KNN, k)
        KNN.fit(KNN, visualizer.Xvar, visualizer.Yvar)

        
        data, classes = load_iris(return_X_y=True)
        
        #Graphic  
        
        #Df = el dataset que carguen  
        df = px.data.iris()
        df = df[["sepal_width", "sepal_length", "species"]]
        fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
        #fig.show()
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
        visualizer.connect_dots(vecinos, classData, visualizer.Xvar, fig2)

  
    def connect_dots(neighbours_position, new_dot, main_matrix, fig2):        
        dot_x = new_dot[1]
        dot_y = new_dot[0] 
        
        
        fig2.show()
        #Tenemos la posición, hay que tomarlo del arreglo con todos los datos para poder conectarlos
        
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
    data = visualizer.askData()
    visualizer.addDot(df, data[0], data[1])
    
    

    