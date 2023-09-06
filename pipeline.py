import os
import pandas as pd
import datetime
import numpy as np
import pandasql as ps
import matplotlib.pyplot as plt
import pixiedust
import sys
sys.path.append('..')
import util
import etl
import pyarrow.parquet as pq
import pyarrow as pa
from functools import reduce  # For Python 3.x
from pyspark.sql import DataFrame
from pyspark.sql.functions import *

def extractdata( spark, icdlist, icdlikelist, outputfolder,prefix="pipeline_",stage=0):
    outputpath = 'file:/home/z_han/work/Oklahoma State/' + outputfolder
    cohortname = prefix+"cohort.parquet"
    cohort_demo_name = prefix+"demo.parquet"
    cohort_lab_name = prefix+"lab.parquet"
    cohort_como_name = prefix+"como.parquet"
    labcodelist = ['4548-4','38483-4','74774-1','718-7','4544-3','49765-1','3043-7','3043-7','77138-6','28539-5','34548-8',
'28540-3','787-2','1751-7','34543-9','41276-7','27344-1','16325-3','16325-3','789-8','6690-2']
    if stage == 0:
        cohort = etl.extract_pt_from_condition_earliest(spark, icdlist=icdlist, icdlikelist=icdlikelist,outputfilename=cohortname, outputfolder = outputfolder)
        print('cohort extracted!')
        stage+=1
        
    cohort = spark.read.parquet(outputpath+'/'+cohortname)
    
    if stage == 1:
        cohort_demo = etl.extractDemo(spark,cohort,outputfilename=cohort_demo_name, outputfolder =outputfolder)
        print('demo extracted!')
        stage+=1
    
    if stage == 2:
        cohort_lab = etl.extractLab(spark,cohort,labcodelist,outputfilename=cohort_lab_name, outputfolder = outputfolder)
        print('lab extracted!')
        stage+=1
        
    if stage == 3:
        cohort_como = etl.extractComorbidity(spark, cohort, exclude_icdlist =icdlist, exclude_icdlikelist = icdlikelist, outputfilename=cohort_como_name, outputfolder = outputfolder)
        print('comorbidity extracted!')
#     raw = [cohort, cohort_demo, cohort_lab, cohort_como]
#     return raw
          
def processdata(spark, icdlist, icdlikelist, outputfolder, topn = 20,prefix = "pipeline_",stage=0):
    outputpath = 'file:/home/z_han/work/Oklahoma State/' + outputfolder
    cohort = spark.read.parquet(outputpath+'/'+prefix+'cohort.parquet')
    cohort_demo = spark.read.parquet(outputpath+'/'+prefix+'demo.parquet')
    cohort_lab = spark.read.parquet(outputpath+'/'+prefix+'lab.parquet')
    cohort_como = spark.read.parquet(outputpath+'/'+prefix+'como.parquet')
    if stage == 0:
        print('/////////////////////////////////////')
        print('Stage 0: processing demographics......')
        outputfilename = prefix+'_demoprocessed.parquet'
        if os.path.exists(outputpath+'/'+outputfilename):
            print('Demographics has already been processed.Moving to next stage...')
            stage += 1
        else:
            demo_processed = etl.process_demo(spark, cohort, cohort_demo)
            stage += 1
            demo_processed.write.mode('overwrite').parquet(outputpath+'/'+outputfilename)
#     df.cache()
#         os.system("hadoop fs -copyToLocal " + outputfilename + " ~/work/Oklahoma%20State/" + outputfolder)
            print('demographics processed!')
    if stage == 1:
        print('/////////////////////////////////////')
        print('Stage 1: processing comorbidities......')
        outputfilename = prefix+'_comoprocessed.parquet'
        if os.path.exists(outputpath+'/'+outputfilename):
            print('Comorbidity has already been processed.Moving to next stage...')
            stage += 1
        else:
            top_como_dict = etl.get_top_comorbidity(spark, cohort_como, topn = topn)
            top_como_dict = etl.filter_top_comorbidity(top_como_dict, removelist = [])
            como_processed = etl.process_como(spark, cohort, cohort_como, topn, top_como_dict = "")
            stage += 1
            como_processed.write.mode('overwrite').parquet(outputfilename)
#     df.cache()
            os.system("hadoop fs -copyToLocal " + outputfilename + " ~/work/Oklahoma%20State/" + outputfolder)
            print('comorbidity processed!')
    if stage == 2:
        print('/////////////////////////////////////')
        print('Stage 2: processing lab......')
        outputfilename = prefix+'_labprocessed.parquet'
        if os.path.exists(outputpath+'/'+outputfilename):
            print('Lab has already been processed.Moving to next stage...')
            stage += 1
        else:
            start = 3
            end = 0
            lab_processed = etl.process_lab_spark(spark, cohort, cohort_lab, start, end)
            outputfilename = prefix+'_labprocessed.parquet'
            lab_processed.write.mode('overwrite').parquet(outputpath+'/'+outputfilename)
    #     df.cache()
    #         os.system("hadoop fs -copyToLocal " + outputfilename + " ~/work/Oklahoma%20State/" + outputfolder)
            print('lab processed!')