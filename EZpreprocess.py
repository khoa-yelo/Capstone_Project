#!/usr/bin/env python
# coding: utf-8

# # **GENERAL PIPELINE**
# 
# DataFrame -> RowSelector -> Encoder -> Imputer -> Scaler -> FeatureSelector and/or DimReductor

# # **RowSelector (Luke)**

# In[1]:


import pandas as pd 
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn import preprocessing as pre
from sklearn.decomposition import PCA, KernelPCA, NMF, FastICA


# In[ ]:


class RowSelector():
    def __init__(self, df: "Input DataFrame", response_column, random_state):
        self.response_column = response_column
        self.df = df       #pd.DataFrame 
        self.random_state = random_state
    def train_test_split(self, test_size: "percent test set", include_nullresponse =  True):
        
        if (include_nullresponse is True):

            df_notNA = self.df[self.df[self.response_column].notnull()]

            df_NA = self.df[self.df[self.response_column].isna()]

            shuffle_df = shuffle(df_notNA, random_state = self.random_state)
          
            test_size_num = int(test_size*len(shuffle_df))

            test_df = shuffle_df[:test_size_num]
            train_df = shuffle_df[test_size_num:]

            train_df = pd.concat([train_df, df_NA])
            
            train_df = shuffle(train_df, random_state = self.random_state)
            test_df = shuffle(test_df, random_state = self.random_state)
            
        else:
            df_notNA = df[df[self.response_column].notnull()]

            X = df_notNA.iloc[:,1:]
            Y = df_notNA.iloc[:,0]
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state=self.random_state)

            train_df = pd.concat([Y_train, X_train], axis=1)
            test_df = pd.concat([Y_test, X_test], axis=1)
            
            train_df = shuffle(train_df, random_state = self.random_state)
            test_df = shuffle(test_df, random_state = self.random_state)

        return train_df, test_df

        """
        
        Design requirements:
        Test rows must not have missing values at response variable, allow missing values for other predictors.
        Split must always be the same 
        
        RETURN train_df, test_df as pd.DataFame 
        
        """
    
    def kfold_split(self, n_fold, df: "Train DataFrame for CV"):
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_fold)

        out = []

        for train_index, test_index in kf.split(df):
            
            test_df = df.iloc[test_index]
            train_df = df.iloc[train_index]
            
            train_df = pd.concat([train_df, test_df[test_df[self.response_column].isna()]])
            test_df = test_df[test_df[self.response_column].notnull()]

            out.append((train_df, test_df))
        
        return out
        

        """
       
       Design requirements:
        Test rows must not have missing values at response variable, allow missing values for other predictors  
        Split must always be the same 
        
        RETURN a list contains n tuples (n == n_fold) with the form:
      
        [(Train_split_1, Val_split_1),
        (Train_split_2, Val_split_2)        
        ]
        
        train_df, test_df as pd.DataFame
        
        """

    
    # Usage examples
    """
    >>> myRowSelector =  RowSelector(myDataFrame)
    >>> train_df, test_df = myRowselector.train_test_split(0.2)
    >>> myFolds = myRowselector.kfold_split(5, train_df)
    >>> for i in myFolds:
    >>>    train_df, val_df  = i
        
            # Fit model and so on 
    
    """    





# # **Encoder (Erika)**

# In[ ]:



class Encoder():
    options = ['Label', 'OneHot'] 
  #entire df will be encoded in one way for now - consider customization later
    def __init__(self, option):
  
        self.option = option

        
        if self.option == "Label":
            self.encoder = OrdinalEncoder()
        elif self.option == "OneHot":
            self.encoder = OneHotEncoder(drop = "first")
            
    def get_cat_columns(self):
        return self.col
    
    def fit(self, df):
        self.col = list(df.select_dtypes(include=['O']).columns)
        new_df = deepcopy(df)
        for c in self.col:
            new_df[c] = new_df[c].fillna(new_df[c].value_counts().index[0]) #impute most common value
        if self.option == 'Label':
             self.encoder.fit(new_df[self.col])
        if self.option == 'OneHot':
            self.encoder.fit(new_df[self.col])

    def transform(self, df):
        new_df = deepcopy(df)
        for c in self.col:
            new_df[c] = new_df[c].fillna(new_df[c].value_counts().index[0]) #impute most common value
        if self.option == 'Label':
            encoded_cols = self.encoder.transform(new_df[self.col])
            encoded_cols = pd.DataFrame(encoded_cols, columns = self.col, index=new_df.index)
        if self.option == 'OneHot':
            encoded_cols = self.encoder.transform(new_df[self.col]).toarray()
            encoded_cols = pd.DataFrame(encoded_cols, columns = self.encoder.get_feature_names(self.col), index=df.index)
        
        new_df = new_df.drop(axis = 1, labels = self.col)

        return pd.concat([new_df, encoded_cols], axis = 1)
  
    def fit_transform(self, df):
        self.col = list(df.select_dtypes(include=['O']).columns)
        new_df = deepcopy(df)
        for c in self.col:
            new_df[c] = new_df[c].fillna(new_df[c].value_counts().index[0]) #impute most common value
        if self.option == 'OneHot':
            return pd.get_dummies(data=new_df, columns=self.col)
        if self.option == 'Label':
            for c in self.col:
                new_df[c] = LabelEncoder().fit_transform(new_df[c])
            return new_df

    def showOptions(cls):
        """
        print out names of all encoders
        """
        print(cls.options)


# 
# # **Imputing (Christopher)**

# In[ ]:



class Imputer():
    #All items MUST be numerical, whether encoded or one hot.
    options = ['DropNa','Mean','Median', 'k-NN', 'MICE']  # names of different imputer
    def __init__(self, option = 'DropNa'):
        #Default imputer is dropping missing values.
        # determine imputer based on option 
        self.option = option
        if option == 'k-NN':
            self.imputer = KNNImputer()
        if option == 'MICE':
            self.imputer = IterativeImputer()
        if option == 'Mean':
            self.imputer = SimpleImputer(strategy = 'mean')
        if option == 'Median':
            self.imputer = SimpleImputer(strategy = 'median')

    def change_option(self, option):
        self.__init__(option)
    def fit(self, df: "Input pd.Dataframe"):
        """
        RETURN: None, fit data to imputer 
        """
        if self.option == 'DropNa':
            return
        self.imputer.fit(df)
        return
    
    def transform(self, df: "Input pd.Dataframe", val = False):
        if self.option == 'DropNa':
            if val == True:
                return pd.DataFrame(SimpleImputer(strategy = 'mean').fit_transform(df), columns = df.columns)
            temp = df.copy()
            missing = df[df.isnull().any(axis=1)].index
            return temp.drop(missing) 
        return pd.DataFrame(self.imputer.transform(df), columns = df.columns)
        """
        RETURN: pd.DataFrame with imputed values 
        """

    @classmethod
    def showOptions(cls):
        """
        print out names of all imputers 
        """
        print(cls.options)
     
    # Usage examples
    
    """
    >>> Imputer.showOptions()
    ["DropNa", "Mean", "k-NN", "MICE"]
    >>> myImputer =  Imputer("k-NN")
    >>> myImputer.fit(Train_df)
    >>> imputed_Train_df = myImputer.transform(Train_df)
    >>> imputed_Test_df = myImputer.transform(Test_df)
    
    """

        
        


# # **Scaler (Christopher)**

# In[ ]:


class Scaler():
    options = ['NoScale', 'Standard', 'MinMax']
    def __init__(self, option = 'NoScale'):
        
        # determine scaler based on option 
        self.option = option
        if option == 'Standard':
            self.scaler = pre.StandardScaler()
        if option == 'MinMax':
            self.scaler = pre.MinMaxScaler()
    def change_option(self, option):
        self.__init__(option)
    def fit(self, df: "Input pd.Dataframe"):
        if self.option == 'NoScale':
            return
        self.scaler.fit(df)
        return
        """
        RETURN: None, fit data to imputer 
        """
    
    def transform(self, df: "Input pd.Dataframe"):
        if self.option == 'NoScale':
            return df
        return pd.DataFrame(self.scaler.transform(df), columns = df.columns, index=df.index)
        """
        RETURN: pd.DataFrame with scaled values 
        """
        
    @classmethod
    def showOptions(cls):
        """
        print out names of all sclalers 
        """
        print(cls.options)
     
    # Usage examples
    
    """
    >>> Scaler.showOptions()
    ["NoScale","Standard", "MinMax"]
    >>> myScaler = Scaler("MinMax")
    >>> myScaler.fit(X_Train_df)
    >>> scaled_Train_df = myScaler.transform(X_Train_df)
    >>> scaled_Test_df = myScaler.transform(X_Test_df)
    
    """


# # **Feature Selection (Frank)**

# In[ ]:


class FeatureSelector:
    options = ['All']
    
    def __init__(self, option = "All"):
         
        self.option = option
#         self.featureSelector = 

    
    def fit(self, df: "Input pd.Dataframe"):
        
        if self.option == 'All':
            return 
        
        """
        RETURN: None, fit data to selector 
        """
    
    def transform(self, df: "Input pd.Dataframe"):
        
        if self.option == 'All':
            return df
        
        """
        RETURN: pd.DataFrame with only selected columns 
        """



    @classmethod
    def showOptions(cls):
        
        """
        print out names of all selectors  
        """
        
        print("Chi^2,Forward,Backward, Recursive Feature Elimination, Ridge Regression")
        print(cls.options)
     
    # Usage examples
    
    
    
    """
    >>> FeatureSelector.showOptions()
    ["NoSelection","chi2", "varianceThreshold, "]
    >>> mySelector =  FeatureSelector("Selector1")
    >>> mySelector.fit(X_Train_df)
    >>> selected_Train_df = mySelector.transform(X_Train_df)
    >>> selected_Test_df = mySelector.transform(X_Test_df)

    """


# # **DimReductor** (Luke)

# In[25]:



class DimReductor:
    
    options = ["Skip", "PCA", "KernelPCA", "NMF", "ICA"]
    
    def __init__(self, option = "Skip"):
        
        # determine selector based on option 
        self.option = option
        
        if option == 'PCA':
            self.dimReductor = PCA()
        if option == 'KernelPCA':
            self.dimReductor = KernelPCA(kernel='rbf')
        if option == 'NMF':
            self.dimReductor = NMF()
        if option == 'ICA':
            self.dimReductor = FastICA()
    
    def fit(self, df: "Input pd.Dataframe"):
        
        if self.option == 'Skip':
            return
        return self.dimReductor.fit(df)
        """
        RETURN: None, fit data to reductor 
        """
    
    def transform(self, df: "Input pd.Dataframe"):
        
        if self.option == 'Skip':
            return df
        return pd.DataFrame(self.dimReductor.transform(df), index=df.index)
        
        """
        RETURN: pd.DataFrame with only reduced columns 
        """
        
    @classmethod
    def showOptions(cls):
        """
        print out names of all reductors  
        """
        print(cls.options)
     
    # Usage examples
    
    """
    >>> DimReductor.showOptions()
    ["NoReduction","PCA", "KernelPCA"]
    >>> myDimReductor =  DimReductor("Selector1")
    >>> myDimReductor.fit(X_Train_df)
    >>> selected_Train_df = myDimReductor.transform(X_Train_df)
    >>> selected_Test_df = myDimReductor.transform(X_Test_df)
    
    """


# # **PreProcessor (Khoa)**

# In[ ]:


class PreProcessor():

    pipeline_format = {
           "Encoder": "[Choice]",
           "Imputer": "[Choice]",
           "Scaler": "[Choice]",
           "FeatureSelection": "[Choice]",
           "DimReduction": "[Choice]"
    }

    def __init__(self, pipeline: "option on each pipe", response_var):
        self.pipeline_dict = pipeline
        
        self.response_var = response_var
        self.encoder = Encoder(self.pipeline_dict["Encoder"])
        self.imputer = Imputer(self.pipeline_dict["Imputer"])
        self.scaler = Scaler(self.pipeline_dict["Scaler"])
        self.featureSelector = FeatureSelector( self.pipeline_dict["FeatureSelection"])
        self.dimReductor = DimReductor(self.pipeline_dict["DimReduction"])
        
        self.pipeline = [self.encoder, self.imputer,                          self.scaler, self.featureSelector, self.dimReductor]
        

    def fit(self, df):
        x = df
        
        for pipe in self.pipeline:
            if type(pipe) == Imputer: 
                pipe.fit(x)
                x = pipe.transform(x)
                x = x.loc[:, x.columns != self.response_var]
            else:  
                pipe.fit(x)
                x = pipe.transform(x)
               

            
    
    def transform(self, df, val = False):
        x = df
        y = []
       
        for pipe in self.pipeline:
            if type(pipe) == Imputer: 
                x = pipe.transform(x, val = val)
                x, y = x.loc[:, x.columns != self.response_var], x[self.response_var]
                         
            else:
                x = pipe.transform(x)  
        return pd.concat([y, x], axis=1)
    
    @classmethod
    
    def get_all_combinations(cls, drop = None):
        all_combinations =  [(e, i,s,f,d)  
                              for e in Encoder.options if e not in drop\
                              for i in Imputer.options if i not in drop\
                              for s in Scaler.options if s not in drop\
                              for f in FeatureSelector.options if f not in drop\
                              for d in DimReductor.options if d not in drop]
        all_pipelines = []
        for i in all_combinations:
            pipe = {"Encoder": i[0],
                    "Imputer": i[1],
                    "Scaler": i[2],
                    "FeatureSelection": i[3],
                    "DimReduction": i[4]}
            all_pipelines.append(pipe)
        return all_pipelines                    
    
    @classmethod
    def show_pipeline_format(cls):
        print(cls.pipeline_format)
