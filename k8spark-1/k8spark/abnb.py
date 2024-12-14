# -*- coding: utf-8 -*-

# Import necessary PySpark libraries
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from k8spark.spark_utils import get_spark
from k8spark import logger
import seaborn as sns

import matplotlib.pyplot as plt
import networkx as nx

__name__ = "abnb"

def main():
    # Define lineage relationships
    lineage_graph = [
        ("Source: Kaggle --> Onedrive", "Transformation"),
        ("Transformation", "Feature Engineering"),
        ("Feature Engineering", "Filtering"),
        ("Filtering", "Destination")
    ]

    # Create the graph
    G = nx.DiGraph()
    G.add_edges_from(lineage_graph)

    # Draw the graph
    plt.figure(figsize=(20, 10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=10000, font_size=12, arrowsize=20)
    plt.title("Data Lineage Flow")
    plt.savefig("datalineage.png")

    # Commented out IPython magic to ensure Python compatibility.
    # %pip install onedrivedownloader
    from onedrivedownloader import download
    import pandas as pd

    download_path = "https://mygsu-my.sharepoint.com/:u:/g/personal/skotian1_gsu_edu/EU2iYLZi-iRGuUd_DLMtyxwBMXgzQDfGbXG9BvJy91y96g?e=I2Tj2b"

    download(download_path, filename="rideshare_kaggle.csv.zip", unzip=True, clean=True)
    pdf = pd.read_csv('rideshare_kaggle.csv')
    #data_directory = "file:/databricks/driver/rideshare_kaggle.csv"
    #print(pdf)
    pdf_sample = pdf.head(2000)  # Take a sample of 1000 rows

    spark = get_spark(name="air_bnb")

    spark.sparkContext.setLogLevel("WARN")
    spark_df = spark.createDataFrame(pdf_sample)
    spark_df.show()

    """#Checking for Outliers using Box-Plots"""

    import seaborn as sns
    import pandas as pd

    # Specify the columns for which you want to create box plots
    boxplot_columns = [
        "price",
        "distance",
        "surge_multiplier",
        "temperature",
        "apparentTemperature",
        "humidity",
        "windSpeed",
        "visibility",
        "cloudCover",
        "temperatureMin",
        "temperatureMax"
    ]

    # Convert PySpark DataFrame to Pandas DataFrame
    pandas_df = spark_df.select(boxplot_columns).toPandas()

    # Ensure columns are numeric and drop non-numeric or NaN values for plotting
    pandas_df = pandas_df.apply(pd.to_numeric, errors="coerce")  # Convert non-numeric to NaN
    pandas_df = pandas_df.dropna()  # Drop rows with NaN values

    # Create box plots for each numeric column
    for column in boxplot_columns:
        if pandas_df[column].dtype in ["float64", "int64"]:  # Ensure the column is numeric
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=pandas_df, y=column, color="skyblue")
            plt.title(f"Box Plot of {column}", fontsize=14)
            plt.ylabel(column, fontsize=12)
            plt.xlabel("Values", fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()
        else:
            print(f"Skipping non-numeric column: {column}")

    from pyspark.sql import functions as F
    # Convert to appropriate types
    spark_df = spark_df.withColumn("price", F.col("price").cast("double"))

    mean_price = spark_df.agg(F.mean("price")).collect()[0][0]
    spark_df = spark_df.na.fill({"price": mean_price})

    # Step 3: Check the max value of price and delete that row
    max_price_row = spark_df.agg(F.max("price")).collect()[0][0]
    spark_df = spark_df.filter(spark_df["price"] != max_price_row)

    # Drop rows where any relevant column is null (if necessary)
    spark_df = spark_df.dropna(subset=['hour', 'month', 'source', 'destination', 'cab_type', 'name', 'price',
                        'distance', 'surge_multiplier', 'day'])

    spark_df.show()


    # Convert datetime to timestamp and extract features
    from pyspark.sql.functions import to_timestamp, dayofweek, when
    spark_df = spark_df.withColumn("datetime", to_timestamp("datetime")) \
        .withColumn("day_of_week", dayofweek("datetime")) \
        .withColumn("is_weekend", when(dayofweek("datetime").isin([1, 7]), 1).otherwise(0))

    # 2. Data Exploration
    spark_df.printSchema()
    spark_df.show()

    import seaborn as sns
    import pandas as pd

    # Convert the Spark DataFrame to a Pandas DataFrame
    #pandas_df = spark_df.toPandas()

    # Use Seaborn to create the histogram
    sns.set(style="whitegrid")  # Set the style for Seaborn
    sns.histplot(data=spark_df.toPandas(), x='price', bins=50, kde=True, color="blue")

    from pyspark.sql.types import NumericType
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.stat import Correlation
    import pandas as pd

    # Drop irrelevant columns and ensure unique names
    columns_to_drop = ['id', 'datetime', 'timezone', 'visibility.1']  # Irrelevant columns
    spark_df_clean = spark_df.drop(*columns_to_drop)

    # Step 1: Ensure only numeric columns
    numeric_cols = [col for col, dtype in spark_df_clean.dtypes if isinstance(spark_df_clean.schema[col].dataType, NumericType)]
    print("Filtered numeric columns: ", numeric_cols)

    # Step 2: Assemble features into a single vector
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features_vector")
    spark_df_vector = assembler.transform(spark_df_clean).select("features_vector")

    # Step 3: Compute Correlation Matrix
    correlation_matrix = Correlation.corr(spark_df_vector, "features_vector").head()[0].toArray()

    # Step 4: Convert and show using Pandas
    corr_df = pd.DataFrame(correlation_matrix, index=numeric_cols, columns=numeric_cols)
    print("Correlation Matrix:\n", corr_df)

    import seaborn as sns

    # Group columns logically (updated to exclude timestamp and uvIndexTime)

    trip_columns = ['price', 'distance', 'surge_multiplier', 'latitude', 'longitude']
    temperature_columns = [
        'temperature', 'apparentTemperature', 'temperatureHigh', 'temperatureLow',
        'apparentTemperatureHigh', 'apparentTemperatureLow', 'dewPoint'
    ]

    precipitation_humidity_columns = [
        'precipIntensity', 'precipProbability', 'humidity', 'cloudCover', 'moonPhase'
    ]

    wind_columns = ['windSpeed', 'windGust', 'windBearing']

    other_weather_columns = ['visibility', 'pressure', 'uvIndex', 'ozone', 'sunriseTime', 'sunsetTime']

    redundant_columns = ['visibility.1', 'icon']

    # Function to plot heatmap for each group
    def plot_group_heatmap(df, group_cols, title):
        valid_cols = [col for col in group_cols if col in df.columns]  # Ensure columns exist in the DataFrame
        if valid_cols:
            subset_corr = df.loc[valid_cols, valid_cols]
            plt.figure(figsize=(10, 8))
            sns.heatmap(subset_corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
            plt.title(title)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig("heatmap_plots.png")

    # Plot heatmaps for each group

    plot_group_heatmap(corr_df, trip_columns, "Correlation Matrix: Trip and Pricing Information")
    plot_group_heatmap(corr_df, temperature_columns, "Correlation Matrix: Temperature Metrics")
    plot_group_heatmap(corr_df, precipitation_humidity_columns, "Correlation Matrix: Precipitation and Humidity")
    plot_group_heatmap(corr_df, wind_columns, "Correlation Matrix: Wind Metrics")
    plot_group_heatmap(corr_df, other_weather_columns, "Correlation Matrix: Other Weather Details")
    plot_group_heatmap(corr_df, redundant_columns, "Correlation Matrix: Redundant Columns")


    spark_df = spark_df.select('hour', 'month', 'source', 'destination', 'cab_type', 'name', 'price',
                        'distance', 'surge_multiplier', 'day', 'is_weekend')

    spark_df.show()

    spark_df.columns

    selected_columns = ['hour', 'month', 'source', 'destination', 'cab_type', 'name', 'price',
                        'distance', 'surge_multiplier', 'day', 'is_weekend']


    from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
    df = spark_df

    # List of categorical columns
    cat_cols = ['source', 'destination', 'cab_type', 'name']

    # Apply StringIndexer only to categorical columns
    for col in cat_cols:
        indexer = StringIndexer(inputCol=col, outputCol=col + "_index")
        df = indexer.fit(df).transform(df)
        df = df.drop(col)

    # Show the result
    df.show()


    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
    from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml import Pipeline


    # Assuming df is your DataFrame
    # Split the data into training and test sets
    train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

    # Define VectorAssembler
    assembler = VectorAssembler(
        inputCols=['hour', 'month', 'source_index', 'destination_index', 'cab_type_index', 'name_index',
                'distance', 'surge_multiplier', 'day', 'is_weekend'],
        outputCol='features'
    )

    #Define Pipeline
    pipeline = Pipeline(stages=[])
    base_pipeline = [assembler]

    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
    from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml import Pipeline

    # Initialize Regression Models
    lr = LinearRegression(labelCol="price", featuresCol="features")
    rf = RandomForestRegressor(labelCol="price", featuresCol="features")
    gbt = GBTRegressor(labelCol="price", featuresCol="features")

    # Define pipelines for each model
    pl_lr = base_pipeline + [lr]
    pl_rf = base_pipeline + [rf]
    pl_gbt = base_pipeline + [gbt]

    # Build parameter grids for each model
    paramGrid_lr = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.1, 0.01]) \
        .addGrid(lr.maxIter, [5, 10]) \
        .baseOn({pipeline.stages: pl_lr}) \
        .build()

    paramGrid_rf = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20]) \
        .addGrid(rf.maxDepth, [3, 5]) \
        .baseOn({pipeline.stages: pl_rf}) \
        .build()

    paramGrid_gbt = ParamGridBuilder() \
        .addGrid(gbt.maxIter, [5, 10]) \
        .addGrid(gbt.maxDepth, [3, 5]) \
        .baseOn({pipeline.stages: pl_gbt}) \
        .build()

    # Merge parameter grids for all models
    paramGrid = paramGrid_lr + paramGrid_rf + paramGrid_gbt

    cv = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=paramGrid,
                        evaluator=RegressionEvaluator(labelCol="price", metricName="r2"),
                        numFolds=3)

    # Fit the cross-validation model
    cvModel = cv.fit(train_data)

    # Evaluate the best model from the cross-validation
    best_model = cvModel.bestModel
    print("Best Model Parameters:", best_model.stages[-1].extractParamMap())

    # Save the model
    #model_path = "file:/databricks/driver/rideshare_model"
    #best_model.write().overwrite().save(model_path)
    print("Model saved successfully")


    best_model_name = best_model.stages[-1].__class__.__name__
    print("Best Model Name:", best_model_name)

    # Test the best model on the test data and evaluate
    test_predictions = best_model.transform(test_data)

    # Evaluate R², RMSE, and MAE
    r2_evaluator = RegressionEvaluator(labelCol="price", metricName="r2")
    rmse_evaluator = RegressionEvaluator(labelCol="price", metricName="rmse")
    mae_evaluator = RegressionEvaluator(labelCol="price", metricName="mae")

    # Calculate R², RMSE, and MAE for the best model on test data
    best_model_r2 = r2_evaluator.evaluate(test_predictions)
    best_rmse_value = rmse_evaluator.evaluate(test_predictions)
    best_mae_value = mae_evaluator.evaluate(test_predictions)

    # Print the evaluation metrics
    print(f"Best Model R2: {best_model_r2}")
    print(f"Best Model Root Mean Squared Error (RMSE): {best_rmse_value}")
    print(f"Best Model Mean Absolute Error (MAE): {best_mae_value}")
