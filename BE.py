from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
"""
import plotly.express as px
import plotly.graph_objects as go
"""


class KNN():

    def __init__(self, k):
        """
        Creates a new KNN object.

        Arguments:
          K: the number of nearest neighboors
        """
        self.k = k
        # Dimensions used (2, x and y)
        self.dimensions = 2
        self.classPosition = 4

    # Recibe el vector a evaluar
    def get_k_nearest_neighboors(self, datapoint):
        """
        Gets the k-nearest neighboors of a given datapoint
        Argunments:
          datapoint: numpy.array, a row vector
        Returns:
          indices: list, indices corresponding with the k datapoints in self.X most
                   similar to datapoint
        """

        distances = []  # distances between the matrix and the datapoint

        size = len(self.data)
        vector_of_the_matrix = []

        for i in range(size):
            vector_of_the_matrix = self.data[i]
            np_vector_of_the_matrix = np.array(vector_of_the_matrix)
            two_vectors_difference = self.calculate_distance(
                self, datapoint, np_vector_of_the_matrix)
            distances.append(two_vectors_difference)

        distances = np.array(distances)
        # Retorna los ínidices de las distancias de menor a mayor
        indices = distances.argsort()
        k_indices = []

        for i in range(self.k):
            k_indices.append(indices[i])

        return k_indices

    # Recibe el vector nuevo que se está evaluando, y el vector de la matriz
    def calculate_distance(self, datapoint1, datapoint2):
        """
        Calculates the euclidean
        Arguments:
          datapoint1: numpy.array, first datapoint. It's the row vector we want to compare with the others.
          datapoint2: numpy.array, second datapoint
        Returns:
          Distance between the given datapoints
        """

        # Parse to x dimesions of the main matrix

        formatedDTP = []

        
        for i in range(self.dimensions):
            formatedDTP.append(datapoint2[i])

        formatedDTP = np.array(formatedDTP)
        if(len(datapoint1) != len(datapoint2)):
              array3 = np.subtract(formatedDTP, datapoint1)
        else:
              array3 = np.subtract(datapoint2, datapoint1)
        return np.linalg.norm(array3)

    #

    def fit(self, main_matrix, y):  # main_matrix antes era X
        """
        Train the model, i.e., allocate the dictionary with features by datapoint 
        and their corresponding class

        Arguments:
          main_matrix: numpy.ndarray, matrix used to train the model, where each row represents a datapoint.  
        No returns:
        """
        self.data = main_matrix
        self.classes = y

    def predict(self, X):
        """
        Predicts the class for each datapoint in the matrix X.
        Arguments:
          X: numpy.ndarray, matrix used to get predictions for each datapoint, where each row represents a datapoint.  
        Returns:
          predictions: numpy.ndarray, class predicted for each datapoint in X
        """
        preds = []
        for datapoint in X:
            index = self.get_k_nearest_neighboors(self, datapoint)
            # Obtener los indices de las clases
            classes = np.array([self.classes[idX] for idX in index])
            # Obtener la clase mas frecuente de los vecinos mas cercanos
            counts = np.bincount(classes)
            # Cuenta cuantos elementos hay de cada número, y los pone en orden (1, 3, 3, 4) --> (1, 0, 2, 1)
            predicted_class = np.argmax(counts)
            # Retorna el índice del mayor valor
            preds.append(predicted_class)
        return np.array(preds)

    def getClass(data):
      pred = []
      
      for i in range(len(data)):
        pred.append(KNN.classes[data[i]])
      pred = np.array(pred) 
      return pred
      
          

if __name__ == '__main__':
  iris = load_iris()
  X = iris.data
  Y = iris.target
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
  data, classes = load_iris(return_X_y=True)
  k = 3
  KNN.__init__(KNN, k)
  KNN.fit(KNN, X, Y)
  newData = [8, 8]
  preds = KNN.get_k_nearest_neighboors(KNN, newData)
  print(X)
  print(Y)
  for i in range(k):
    print(X[preds[i]])
