from pyspark.sql import SparkSession
import transpile_spark as tspk
import conversion_rules as cr
import conv_tracker as ct
import os, json
from datetime import datetime

# save the time in a variable and print the time
print("Current Time =", datetime.now().strftime("%H:%M:%S"))
start = datetime.now()

example_dir = "./examples"
output_dir = "../../../sc_output_files"
#xml_file = "av_bal.XML"
#xml_file = "CmplxPeriod.XML"
#xml_file = "ageAtDate.XML"
#xml_file = "myPandL.XML"
#xml_file = "ranch.XML" #need fix to properly prefix table
#xml_file = "endDateDays.XML"
#xml_file = 'PeriodDiff.XML'
xml_file = 'endDateMonths.XML'

conversion_tracker = ct.initialize_conversion_tracker()
overrides = {}
mode = "build"  #'options:  'build' 'complete' 'supplement'

spark = SparkSession.builder.appName("Test app").getOrCreate()

tspk.test_only(
    spark, xml_file, example_dir, mode, conversion_tracker, overrides
)
print("Total Time = ", datetime.now() - start)
