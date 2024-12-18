import findspark
findspark.init()

from pyspark.ml.classification import GBTClassificationModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import PipelineModel
from pyspark.sql.functions import when, col
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
import pandas as pd

class ModelPredictor:
    def __init__(self, model_path):
        # Initialize Spark Session

        self.spark = SparkSession.builder \
            .appName("PySpark_Prediction_App") \
            .config("spark.driver.host", "localhost") \
            .config("spark.executor.memory", "2g") \
            .master("local[*]") \
            .getOrCreate()

        # Load pre-trained model
        self.model = PipelineModel.load(model_path)
    
    def convert_columns(self,data):
    # Assume data is a Pandas DataFrame
        columns = ['BMI', 'Smoking', 'Stroke', 'PhysicalHealth', 'DiffWalking', 'Sex', 'Diabetic', 'PhysicalActivity', 'GenHealth']
    
        for column in columns:
            # Replace categorical values with numeric
            data[column] = data[column].replace({
                "Yes": 1.0,
                "No": 0.0,
                "Male": 1.0,
                "Female": 0.0,
                "Excellent": 2.0,
                "Very good": 0.0,
                "Good": 1.0,
                "Fair": 3.0,
                "Poor": 4.0
            })

        # Convert columns to float if necessary
        for column in columns:
            data[column] = data[column].astype(float)

        return data
        
    def predict(self, input_data):
        # Convert input to Spark DataFrame
        try:
            schema = StructType([
                StructField("BMI", DoubleType(), True),
                StructField("Smoking", DoubleType(), True),
                StructField("Stroke", DoubleType(), True),
                StructField("PhysicalHealth", DoubleType(), True),
                StructField("DiffWalking", DoubleType(), True),
                StructField("Sex", DoubleType(), True),
                StructField("Diabetic", DoubleType(), True),
                StructField("PhysicalActivity", DoubleType(), True),
                StructField("GenHealth", DoubleType(), True),
            ])
            spark_df = self.spark.createDataFrame(input_data, schema=schema)
            print('spark df')
            print(spark_df.show())
            #Vector assemble data
            #features = ['BMI', 'Smoking', 'Stroke', 'PhysicalHealth', 'DiffWalking', 'Sex', 'Diabetic', 'PhysicalActivity', 'GenHealth']
            #assembler = VectorAssembler(inputCols=features, outputCol="features")
            #spark_df = assembler.transform(spark_df)

            # Make predictions
            predictions = self.model.transform(spark_df)
            print(predictions)
            # Convert predictions to pandas for easier handling
            return predictions.toPandas()
        except Exception as e:
            raise ValueError(f"Lỗi khi dự đoán: {str(e)}")
    def close(self):
        # Stop Spark session
        self.spark.stop()