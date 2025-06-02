import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomExecption
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_obj(self):
        #this function is responsible for data transformation
        
        try:
            numerical_coloumns = ['reading_score', 'writing_score']
            categorical_coloumns =  ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_pipeline = Pipeline(
                steps=[
                    ("imuter",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            cat_pipleline = Pipeline(
                steps= [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoder",OneHotEncoder()),
                    ("standard_scaling", StandardScaler(with_mean=False))
                ]
            )
            logging.info("numerical colomns standard scaling completed")
            
            logging.info("categorical colomns encoding completed")
            
            preprocessor =ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_coloumns),
                    ("cat_piepline",cat_pipleline,categorical_coloumns)
                    
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomExecption(e,sys)    
            
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("read train and test data completed")
            logging.info("obtianing preprocessing object")
            
            preprocessor_obj = self.get_data_transformer_obj()
            
            target_coloumn_name = "math_score"
            numerical_coloumns = [ 'reading_score', 'writing_score']
            
            input_feature_train_df = train_df.drop(columns=target_coloumn_name,axis=1)
            target_feature_train_df = train_df[target_coloumn_name]
            
            input_feature_test_df = test_df.drop(columns=target_coloumn_name,axis=1)
            target_feature_test_df = test_df[target_coloumn_name]
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info(f"saved preprocessing object.")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            
           
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomExecption(e,sys)        