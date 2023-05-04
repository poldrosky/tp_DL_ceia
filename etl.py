
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


class ETL:
    
    def generate_data(self, dataset):        
        
        #cat_Product_ID = dataset['Product_ID'].unique().tolist()
        cat_Gender = dataset['Gender'].unique().tolist()
        cat_Age = dataset['Age'].unique().tolist()
        cat_City_Category = dataset['City_Category'].unique().tolist()
        cat_Stay_In_Current_City_Years = dataset['Stay_In_Current_City_Years'].unique().tolist()

        self.features_categorical = [
        #    'Product_ID',
            'Gender',
            'Age',
            'City_Category',
            'Stay_In_Current_City_Years'
        ]

        self.values_categorical = [
        #    cat_Product_ID,
            cat_Gender,
            cat_Age,
            cat_City_Category,
            cat_Stay_In_Current_City_Years         
        ]
        
        self.encoder_categorical = self.fit_encoder_categorical(dataset)
        dataset = self.transform_encoder_categorical(dataset)
        
        dataset = self.drop_columns(dataset)
        
        self.mean_Product_Category_2 = dataset.Product_Category_2.mean()
        self.mean_Product_Category_3 = dataset.Product_Category_3.mean()
        
        dataset = self.impute_data(dataset)
        
        
        return dataset
        
        
    def transform_etl(self, df):
        
        df = self.transform_encoder_categorical(df)
        df = self.drop_columns(df)
        df = self.impute_data(df)
        
        return df
    
    def impute_data(self, dataset):
        ''' impute data with mean '''
        dataset.Product_Category_2 = dataset.Product_Category_2.fillna(self.mean_Product_Category_2)
        dataset.Product_Category_3 = dataset.Product_Category_3.fillna(self.mean_Product_Category_3)
        
        return dataset
    
    
    def drop_columns(self, dataset):
        
        columns = self.features_categorical 
        dataset = dataset.drop(columns = columns)
        return dataset        
    
    
    def transform_encoder_categorical(self, dataset):
        ''' transform features categorical to onehotencoder '''
        dataset = pd.concat([
            dataset,
            pd.DataFrame(self.encoder_categorical.transform(
                dataset[self.features_categorical]).toarray(),
                         columns = self.encoder_categorical.get_feature_names_out())
        ], axis = 1)
              
        return dataset
    
    
    def fit_encoder_categorical(self, dataset):
        ''' fit features categorical to onehotencoder '''
        encoder_categorical = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(
                    categories = self.values_categorical, handle_unknown='ignore'), self.features_categorical)
            ])
                 
        encoder_categorical.fit(dataset[self.features_categorical])
                 
        return encoder_categorical    