from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
from datetime import datetime


import os, shutil, time
from dotenv import load_dotenv

load_dotenv()  # Takes .env variables and adds them to the environment
# make sure to have an .env file with PYSPARK_PYTHON set to your python (likely virtual env for development)
pyspark_python = os.environ.get("PYSPARK_PYTHON", None)
if not pyspark_python:
    raise EnvironmentError("The PYSPARK_PYTHON environment variable is not set.")
os.environ["SPARK_LOCAL_DIRS"] = "./spark-temp"


def main() -> None:
    output_dir_base = "./data/spark_output_files"

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_output_dir = f"{output_dir_base}_{timestamp}"

    spark = SparkSession.builder.appName("first app").getOrCreate()
    spark = SparkSession.builder.appName("second app").getOrCreate()

    # main data set.
    data = [
        ("John Doe", 30, 65),
        ("Jane Doe", 25, 60),
        ("Mike Johnson", 60, 65),
        ("Sophia Smith", 64, 65),
    ]
    columns = ["Name", "Age", "Eligible_Retirement_Age"]
    df = spark.createDataFrame(data, schema=columns)

    df = df.withColumn(
        "YearsToRetirement", expr("Greatest(Eligible_Retirement_Age - Age, 0)")
    )

    # Reference dataset
    status_data = [("Gold", 0), ("Silver", 1), ("Bronze", 2)]
    status_columns = ["Status", "YearsToRetirement"]
    status_df = spark.createDataFrame(status_data, schema=status_columns)

    df.createOrReplaceTempView("main")
    status_df.createOrReplaceTempView("status")

    df = spark.sql(
        """
    SELECT 
        m.*, 
        COALESCE(s.Status, 'No Status') as Status
    FROM 
        main m
    LEFT JOIN 
        status s 
    ON 
        m.YearsToRetirement = s.YearsToRetirement
    """
    )

    df = df.withColumn(
        "New_Status",
        expr("case when YearsToRetirement > 5 then 'plan ahead' else Status end"),
    )
    df = df.drop("Status")

    # save to CSV
    output_file = os.path.join(unique_output_dir, "final_dataset.csv")
    df.coalesce(1).write.csv(output_file, mode="overwrite", header=True)


    #line2
    time.sleep(5) 
    spark.stop()
    print("complete")




if __name__ == "__main__":
    main()
