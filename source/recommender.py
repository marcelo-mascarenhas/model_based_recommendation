import numpy as np

import pandas as pd


class Recommender():
    """
        Class that stores some useful methods and attributes to produce the recommendation. 
        After an instance of the class is initialized, it should be trained with the method 'funkSvd', and then it
        can be used to predict the scores of a given item and user. 
    """
    
    def __init__(self, data, number_of_dimensions=5, init_num=1):
        """
        Parameters
        -----------
        
        data => Path to file which containers UserID, ItemIDs and Ratings to train the matrix using FunkSvd.
        number_of_dimensions => Number of latent dimensions.
        init_num => 
        
        
        
        Attributes
        -----------
        __data_matrix => Store the target dataframe, with the current rankings.
        __k => Number of latent dimensions.
        __average_score => mean score of the ratings contained in the given dataset. This is used to substitute the
        scores that are bigger than 5 in the model.
        
        
        """
        self.__data_matrix = self.openDataFrame(data)
        
        self.__k = number_of_dimensions
        
        self.p_matrix_tuple, self.q_matrix_tuple = self.__get_matrixes_tuples(self.__data_matrix, init_num)
        
        self.__average_score = self.__data_matrix['Rating'].mean()
        
        self.__max_score = self.__data_matrix['Rating'].max()
        
    def openDataFrame(self, path):
        df = pd.read_csv(f'{path}', encoding="utf-8", header=None, \
            names=['UserID', 'ItemID', 'Rating'], skiprows=1, sep=':|,', \
                engine='python')
        return df
    
    

    def __get_matrixes_tuples(self, df, standard_value=1):
        """
        Calculate two matrixes, P and Q, in a way that:
        P: |Users| * K: Equivalent to user feature matrix.
        Q: K * |Items|: Equivalent to item feature matrix.
        Also generates two dictionaries that map for items and user matrixes w.r.t their ids..
        
        Returns
        ----------
        p_matrix_tuple = (user_matrix, dictionary)
        q_matrix_tuple = (item_matrix, dictionary)
        

        """
        
        only_users = df['UserID'].unique()
        only_items = df['ItemID'].unique()

        integer_list = []
        for i in range(self.__k):
            integer_list.append(i)

        p_matrix = pd.DataFrame(columns=integer_list, index=only_users).fillna(standard_value)
        
        q_matrix = pd.DataFrame(columns=integer_list, index=only_items).fillna(standard_value)

        p_dc = {v: k for k,v in enumerate(p_matrix.index)}

        q_dc = {v: k for k,v in enumerate(q_matrix.index)}


        p_matrix = (p_matrix.to_numpy()).astype('float128')

        q_matrix = (q_matrix.to_numpy()).astype('float128')

        q_matrix = q_matrix.T

        return (p_matrix, p_dc), (q_matrix, q_dc)


    def funkSvd(self, learning_rate=0.005, epochs=20, beta=0.02):

        p_matrix, p_dc = self.p_matrix_tuple

        q_matrix, q_dc = self.q_matrix_tuple

        for i in range (epochs):
            for uid,iid,rating in self.__data_matrix.to_numpy():

                #Initializing parameters to make the code more readable
                p_vector = p_matrix[p_dc[uid], :]
                q_vector = q_matrix[:, q_dc[iid]]


                error = rating - p_vector @ q_vector

                #Calculate gradient vectors.
                p_grad_vector, q_grad_vector = self.__gradient_vectors(p_vector, q_vector, error)

                #Update matrixes
                q_matrix[:, q_dc[iid]] += learning_rate*(q_grad_vector - beta*q_vector)

                p_matrix[p_dc[uid], :] += learning_rate*(p_grad_vector - beta*p_vector)
            
    def __gradient_vectors(self, pvector, qvector, error):    

        p_gradv = 2*error*qvector

        q_gradv = 2*error*pvector

        return p_gradv, q_gradv
    
    def evaluate(self, target):
        """
        Receive target path which contains the dataframe. The function returns the column with the scores.
        
        """
        p_matrix, p_dc = self.p_matrix_tuple

        q_matrix, q_dc = self.q_matrix_tuple
        
        df = self.openDataFrame(target)
                
        df['Rating'] = df.apply(lambda x: p_matrix[p_dc[x['UserID']], :] @ q_matrix[:, q_dc[x['ItemID']]] \
            if p_matrix[p_dc[x['UserID']], :] @ q_matrix[:, q_dc[x['ItemID']]] <= self.__max_score \
                else self.__average_score, axis=1)
        
        return df['Rating']
    
    def save_csv(self, target_df, rating_column):
        tdf = pd.read_csv(f'{target_df}')
        
        tdf['Rating'] = rating_column
        
        return tdf
