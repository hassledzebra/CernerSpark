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
import pyarrow.parquet as pq
import pyarrow as pa
from functools import reduce  # For Python 3.x
from pyspark.sql import DataFrame
from pyspark.sql.functions import *

from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StandardScaler,OneHotEncoder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer

def extract_pt_from_condition(spark, icdlist=[], icdlikelist=[],outputfilename="", outputfolder = ""):
#     items in icdlist and icdlikes should be single quote mark
# the outputfolder should be like "Zheng_Han/pompe". It is appended within the Oklahoma state folder. No relative path is allowed.

    if outputfilename =="":
        print("please set outputfilename!")
        return None
    
    if outputfolder =="":
        print("please set outputfolder!")
        return None
        
    sql_sentence = "select distinct personid, encounterid, effectivedate as date, conditioncode.standard.id as conditioncode from condition where "
    cnt = 0
    insentence = ""
    likesentence = ""
    
    if len(icdlist) > 0:
        insentence = "conditioncode.standard.id in ("
        for i in range(len(icdlist)-1):
            insentence = insentence + "'" + icdlist[i] + "'" + ","
        insentence = insentence + "'" +icdlist[-1] + "'"
        insentence += ")"
        
        cnt +=1
    
    if len(icdlikelist) > 0:
        likesentence = "conditioncode.standard.id like "
        for i in range(len(icdlikelist)-1):
            likesentence = likesentence + "'" + icdlikelist[i] + "'" 
            likesentence += " or conditioncode.standard.id like "

        likesentence = likesentence + "'" + icdlikelist[-1] +  "'" 
        cnt +=1
        
    if cnt == 2:
        sql_sentence = sql_sentence + insentence + " or " + likesentence
    else:
        sql_sentence = sql_sentence + insentence + likesentence
          
    print(sql_sentence)
    df = spark.sql(sql_sentence)
    df.write.mode('overwrite').parquet(outputfilename)
    df.cache()
    os.system("hadoop fs -copyToLocal " + outputfilename + " ~/work/Oklahoma%20State/" + outputfolder)
    return df

def extract_pt_from_condition_earliest(spark, icdlist=[], icdlikelist=[],outputfilename="", outputfolder = ""):
#     items in icdlist and icdlikes should be single quote mark
# the outputfolder should be like "Zheng_Han/pompe". It is appended within the Oklahoma state folder. No relative path is allowed.

    if outputfilename =="":
        print("please set outputfilename!")
        return None
    
    if outputfolder =="":
        print("please set outputfolder!")
        return None
        
    sql_sentence = "select distinct personid,  min(effectivedate) as date from condition where "
    cnt = 0
    insentence = ""
    likesentence = ""
    
    if len(icdlist) > 0:
        insentence = "conditioncode.standard.id in ("
        for i in range(len(icdlist)-1):
            insentence = insentence + "'" + icdlist[i] + "'" + ","
        insentence = insentence + "'" +icdlist[-1] + "'"
        insentence += ")"
        
        cnt +=1
    
    if len(icdlikelist) > 0:
        likesentence = "conditioncode.standard.id like "
        for i in range(len(icdlikelist)-1):
            likesentence = likesentence + "'" + icdlikelist[i] + "'" 
            likesentence += " or conditioncode.standard.id like "

        likesentence = likesentence + "'" + icdlikelist[-1] +  "'" 
        cnt +=1
        
    groupbysentence = " group by personid"
        
    if cnt == 2:
        sql_sentence = sql_sentence + insentence + " or " + likesentence + groupbysentence
    else:
        sql_sentence = sql_sentence + insentence + likesentence + groupbysentence
          
    print(sql_sentence)
    df = spark.sql(sql_sentence)
    df.write.mode('overwrite').parquet(outputfilename)
    df.cache()
    os.system("hadoop fs -copyToLocal " + outputfilename + " ~/work/Oklahoma%20State/" + outputfolder)
    return df


def extract_pt_from_procedure(spark, cptlist=[], cptlikelist=[],outputfilename="", outputfolder = ""):
#     items in icdlist and icdlikes should be single quote mark
# the outputfolder should be like "Zheng_Han/pompe". It is appended within the Oklahoma state folder. No relative path is allowed.

    if outputfilename =="":
        print("please set outputfilename!")
        return None
    
    if outputfolder =="":
        print("please set outputfolder!")
        return None
    
            
    sql_sentence = "select distinct personid, encounterid, servicestartdate as date, procedurecode.standard.id as procedurecode from procedure where "
    cnt = 0
    insentence = ""
    likesentence = ""
    
    if len(cptlist) > 0:
        insentence = "procedurecode.standard.id in ("
        for i in range(len(cptlist)-1):
            insentence = insentence + "'" + cptlist[i] + "'" + ","
        insentence = insentence + "'" +cptlist[-1] + "'"
        insentence += ")"
        
        cnt +=1
    
    if len(cptlikelist) > 0:
        likesentence = "procedurecode.standard.id like "
        for i in range(len(cptlikelist)-1):
            likesentence = likesentence + "'" + cptlikelist[i] + "'" 
            likesentence += " or procedurecode.standard.id like "

        likesentence = likesentence + "'" + cptlikelist[-1] +  "'" 
        cnt +=1
#     groupbysentence = " group by personid"
    
    if cnt == 2:
        sql_sentence = sql_sentence + insentence + " or " + likesentence 
    else:
        sql_sentence = sql_sentence + insentence + likesentence 
          
    print(sql_sentence)
    df = spark.sql(sql_sentence)
    df.write.mode('overwrite').parquet(outputfilename)
    df.cache()
    os.system("hadoop fs -copyToLocal " + outputfilename + " ~/work/Oklahoma%20State/" + outputfolder)
    return df

def extract_pt_from_procedure_earliest(spark, cptlist=[], cptlikelist=[],outputfilename="", outputfolder = ""):
#     items in icdlist and icdlikes should be single quote mark
# the outputfolder should be like "Zheng_Han/pompe". It is appended within the Oklahoma state folder. No relative path is allowed.

    if outputfilename =="":
        print("please set outputfilename!")
        return None
    
    if outputfolder =="":
        print("please set outputfolder!")
        return None
    
            
    sql_sentence = "select distinct personid, encounterid, servicestartdate as date, procedurecode.standard.id as procedurecode from procedure where "
    cnt = 0
    insentence = ""
    likesentence = ""
    
    if len(cptlist) > 0:
        insentence = "procedurecode.standard.id in ("
        for i in range(len(cptlist)-1):
            insentence = insentence + "'" + cptlist[i] + "'" + ","
        insentence = insentence + "'" +cptlist[-1] + "'"
        insentence += ")"
        
        cnt +=1
    
    if len(cptlikelist) > 0:
        likesentence = "procedurecode.standard.id like "
        for i in range(len(cptlikelist)-1):
            likesentence = likesentence + "'" + cptlikelist[i] + "'" 
            likesentence += " or procedurecode.standard.id like "

        likesentence = likesentence + "'" + cptlikelist[-1] +  "'" 
        cnt +=1
    
    if cnt == 2:
        sql_sentence = sql_sentence + insentence + " or " + likesentence
    else:
        sql_sentence = sql_sentence + insentence + likesentence
          
    print(sql_sentence)
    df = spark.sql(sql_sentence)
    df.write.mode('overwrite').parquet(outputfilename)
    df.cache()
    os.system("hadoop fs -copyToLocal " + outputfilename + " ~/work/Oklahoma%20State/" + outputfolder)
    return df

def extract_pt_from_med(spark, druglikelist=[],outputfilename="", outputfolder = ""):
#     items in icdlist and icdlikes should be single quote mark
# the outputfolder should be like "Zheng_Han/pompe". It is appended within the Oklahoma state folder. No relative path is allowed.

    if outputfilename =="":
        print("please set outputfilename!")
        return None
    
    if outputfolder =="":
        print("please set outputfolder!")
        return None
    
            
    sql_sentence = "select distinct encounterid, personid, drugcode.standard.id as drugid, drugcode.standard.primaryDisplay as drugname, startdate, stopdate from medication where "

    likesentence = ""
    
    
    
    if len(druglikelist) > 0:
        likesentence = "drugcode.standard.primaryDisplay like "
        for i in range(len(druglikelist)-1):
            likesentence = likesentence + "'%" + druglikelist[i] + "%'" 
            likesentence += " or drugcode.standard.primaryDisplay like "

        likesentence = likesentence + "'%" + druglikelist[-1] +  "%'" 
        
   
    sql_sentence = sql_sentence + likesentence
          
    print(sql_sentence)
    df = spark.sql(sql_sentence)
    df.write.mode('overwrite').parquet(outputfilename)
    df.cache()
    os.system("hadoop fs -copyToLocal " + outputfilename + " ~/work/Oklahoma%20State/" + outputfolder)
    return df

def cohort_statistics(spark, sparkdf):
    print("The number of unique patients: ",sparkdf.select('personid').distinct().count())
    print("The number of unique encounters: ",sparkdf.select('encounterid').distinct().count())
    


def mergeCohort(spark, pqdfs, outputfilename="", outputfolder=""):
    if outputfilename =="":
        print("please set outputfilename!")
        return None
    
    if outputfolder =="":
        print("please set outputfolder!")
        return None
#     df = reduce(DataFrame.unionAll, sparkdfs)
    pddfs = [x.to_pandas() for x in pqdfs]
    df = pd.concat(pddfs)
    df = spark.createDataFrame(df)
    df.write.mode('overwrite').parquet(outputfilename)
    df.cache()
    os.system("hadoop fs -copyToLocal " + outputfilename + " ~/work/Oklahoma%20State/" + outputfolder)
    return df

labcodelist = ['4548-4','38483-4','74774-1','718-7','4544-3','49765-1','3043-7','3043-7','77138-6','28539-5','34548-8',
'28540-3','787-2','1751-7','34543-9','41276-7','27344-1','16325-3','16325-3','789-8','6690-2']  

def extractLab(spark,cohort,labcodelist=labcodelist,outputfilename="", outputfolder = ""):
    if outputfilename =="":
        print("please set outputfilename!")
        return None
    
    if outputfolder =="":
        print("please set outputfolder!")
        return None
    
    cohort.createOrReplaceTempView('cohort')
#     sql_sentence = "select distinct l.labid, l.encounterid, l.personid, l.labcode.standard.id as labcode, "
#     sql_sentence += "l.labcode.standard.primaryDisplay, l.loincclass, l.servicedate, l.typedvalue.numericValue.value as value, "
#     sql_sentence += "l.interpretation.standard.primaryDisplay as interpretation,l.source from lab l inner join cohort r "
#     sql_sentence += "on l.personid = r.personid where "
    
    sql_sentence = "select distinct l.labid, l.encounterid, l.personid, r.date, l.labcode.standard.id as labcode, "
    sql_sentence += "l.servicedate, l.typedvalue.numericValue.value as value "
    sql_sentence += "from lab l inner join cohort r "
    sql_sentence += "on l.personid = r.personid where "
    

    insentence = "l.labcode.standard.id in ("
    for i in range(len(labcodelist)-1):
        insentence = insentence + "'" + labcodelist[i] + "'" + ","
    insentence = insentence + "'" +labcodelist[-1] + "'"
    insentence += ")"
    
    sql_sentence = sql_sentence + insentence
    print(sql_sentence)
    df = spark.sql(sql_sentence)
    df.write.mode('overwrite').parquet(outputfilename)
    df.cache()
    os.system("hadoop fs -copyToLocal " + outputfilename + " ~/work/Oklahoma%20State/" + outputfolder)
    return df


def extractDemo(spark,cohort,outputfilename="", outputfolder = ""):
    if outputfilename =="":
        print("please set outputfilename!")
        return None
    
    if outputfolder =="":
        print("please set outputfolder!")
        return None
    
    cohort.createOrReplaceTempView('cohort')
    sql_sentence = "select distinct l.personid, l.birthdate, l.gender.value as gender, "
    sql_sentence += "l.race.value as race from "
    sql_sentence += "dedupedemographics l inner join cohort r "
    sql_sentence += "on l.personid = r.personid"
    print(sql_sentence)
    df = spark.sql(sql_sentence)
    df.write.mode('overwrite').parquet(outputfilename)
#     df.cache()
    os.system("hadoop fs -copyToLocal " + outputfilename + " ~/work/Oklahoma%20State/" + outputfolder)
    return df

def extractComorbidity(spark, cohort, interval = 0, exclude_icdlist =[], exclude_icdlikelist = [], outputfilename="", outputfolder = ""):
#     items in icdlist and icdlikes should be single quote mark
# the outputfolder should be like "Zheng_Han/pompe". It is appended within the Oklahoma state folder. No relative path is allowed.

    if outputfilename =="":
        print("please set outputfilename!")
        return None
    
    if outputfolder =="":
        print("please set outputfolder!")
        return None
    
    cohort.createOrReplaceTempView('cohort')
    
    insentence = ""
    likesentence = ""
        
    sql_sentence = "select distinct l.personid, l.encounterid, l.conditioncode.standard.id as comorbidityid, "
    sql_sentence += "l.effectivedate "
    sql_sentence += "from condition l inner join cohort r on l.personid = r.personid where datediff(l.effectivedate, r.date) >= " + str(interval) + " "
    
    cnt=0
    
    if len(exclude_icdlist) > 0:
        insentence = "l.conditioncode.standard.id not in ("
        for i in range(len(exclude_icdlist)-1):
            insentence = insentence + "'" + exclude_icdlist[i] + "'" + ","
        insentence = insentence + "'" +exclude_icdlist[-1] + "'"
        insentence += ")"
        
        cnt +=1
    
    if len(exclude_icdlikelist) > 0:
        likesentence = "l.conditioncode.standard.id not like "
        for i in range(len(exclude_icdlikelist)-1):
            likesentence = likesentence + "'" + exclude_icdlikelist[i] + "'" 
            likesentence += " and conditioncode.standard.id not like "

        likesentence = likesentence + "'" + exclude_icdlikelist[-1] +  "'" 
        cnt +=1
        
    if cnt == 2:
        sql_sentence = sql_sentence + "and  " + insentence + " and " + likesentence
    elif cnt == 0:
        sql_sentence = sql_sentence
    else:
        sql_sentence = sql_sentence + "and " + insentence + likesentence
          
          
    print(sql_sentence)
    df = spark.sql(sql_sentence)
    df.write.mode('overwrite').parquet(outputfilename)
#     df.cache()
    os.system("hadoop fs -copyToLocal " + outputfilename + " ~/work/Oklahoma%20State/" + outputfolder)
    return df

def get_top_comorbidity(spark, como, topn):
    top_como = como.groupBy('comorbidityid').count().orderBy(col('count').desc())\
        .select(collect_list('comorbidityid')).collect()[0]\
            .__getitem__('collect_list(comorbidityid)')[0:topn]
            
    top_como_coded = ["n"+x.replace('.','_') for x in top_como]
    out = {top_como[i]:top_como_coded[i] for i in range(len(top_como))} # store in a dictionary
    print(out)
    return out
    
def filter_top_comorbidity(top_como_dict, removelist):
    out = top_como_dict
    for topop in removelist:
        out.pop(topop)
    return out

def process_demo(spark, cohort, demo):
    demo.createOrReplaceTempView('demo_temp')
    cohort.createOrReplaceTempView('cohort_temp')    
    
    demo_processed = spark.sql("""
    select distinct l.personid, 
        CASE
            WHEN l.birthdate.value <> '' THEN l.birthdate.value
            ELSE NULL
        END AS birthdate,
        CASE
            WHEN l.gender = 'Male' THEN 'Male'
            WHEN l.gender = 'Female' THEN 'Female'
            WHEN l.gender = 'None' THEN Null
            ELSE 'other_gender'
        END AS gender,
        CASE
            WHEN l.race = 'White' THEN 'White'
            WHEN l.race in ('Black','African American') Then 'Black'
            WHEN l.race = 'Asian' Then 'Asian'
            WHEN l.race IN ('Hispanic, black', 'Hispanic, white', 'Hispanic') THEN 'Hispanic'
            WHEN l.race = 'Native American' THEN 'Native'
            ELSE 'Other_race'
        END AS race,
        '1' AS conditioncode,
        r.date as diagnosis_date
    FROM demo_temp l
    inner join cohort_temp r
    on l.personid = r.personid
    order by l.personid
    """)
    demo_processed = demo_processed.withColumn('age_at_diagnosis',lit(months_between(col('diagnosis_date'),col('birthdate'))/12))
    demo_processed.cache()
      
    return demo_processed

def process_control_demo(spark, cohort, demo):
    demo.createOrReplaceTempView('demo_temp')
    cohort.createOrReplaceTempView('cohort_temp')    
    
    demo_processed = spark.sql("""
    select distinct l.personid, 
        CASE
            WHEN l.birthdate.value <> '' THEN l.birthdate.value
            ELSE NULL
        END AS birthdate,
        CASE
            WHEN l.gender = 'Male' THEN 'Male'
            WHEN l.gender = 'Female' THEN 'Female'
            WHEN l.gender = 'None' THEN Null
            ELSE 'other_gender'
        END AS gender,
        CASE
            WHEN l.race = 'White' THEN 'White'
            WHEN l.race in ('Black','African American') Then 'Black'
            WHEN l.race = 'Asian' Then 'Asian'
            WHEN l.race IN ('Hispanic, black', 'Hispanic, white', 'Hispanic') THEN 'Hispanic'
            WHEN l.race = 'Native American' THEN 'Native'
            ELSE 'Other_race'
        END AS race,
        '0' AS conditioncode,
        r.date as diagnosis_date
    FROM demo_temp l
    inner join cohort_temp r
    on l.personid = r.personid
    order by l.personid
    """)
    demo_processed = demo_processed.withColumn('age_at_diagnosis',lit(months_between(col('diagnosis_date'),col('birthdate'))/12))
    demo_processed.cache()
      
    return demo_processed

def process_como(spark, cohort, comorbidity, topn, top_como_dict = "", count = False, indexdate = 0):
    #     e.g. indexdate = 6: comorbidities of 6 months before diagnosis
    cohort.createOrReplaceTempView('cohort_temp')
#     comorbidity.createOrReplaceTempView('comorbidity_temp')
    
    joined = comorbidity.alias('l').select('personid','comorbidityid','effectivedate').join(cohort.alias('c'), 'personid')
    joined_filtered = joined.filter((months_between(col('date'),col('effectivedate')) >= indexdate))
    
    joined_filtered.createOrReplaceTempView('comorbidity_temp')
    
    if top_como_dict == "":
        top_comorbidity = get_top_comorbidity(spark, comorbidity, topn)
    else:
        top_comorbidity = top_como_dict
        
    if count:
        sql_como = "select personid,count(case when comorbidityid = "
        for i, (k, v) in enumerate(top_comorbidity.items()):
            if i < len(top_comorbidity) - 1:
                sql_como = sql_como + "'" + k + "'" + " then 1 end) as "+ v + ", count(case when comorbidityid = "
            else:
                sql_como = sql_como + "'" + k + "' then 1 end) as " + v + " "
        
    else:    
        sql_como = "select personid,count(case when comorbidityid = "
        for i, (k, v) in enumerate(top_comorbidity.items()):
            if i < len(top_comorbidity) - 1:
                sql_como = sql_como + "'" + k + "'" + " then 1 end) > 0 as "+ v + ", count(case when comorbidityid = "
            else:
                sql_como = sql_como + "'" + k + "' then 1 end) > 0 as " + v + " "
                
    sql_como = sql_como + "from comorbidity_temp where comorbidityid in ("
    
    for i, (k, v) in enumerate(top_comorbidity.items()):
        if i < len(top_comorbidity) - 1:
            sql_como = sql_como + "'" + k + "'" +  ", "
        else:
            sql_como = sql_como + "'" + k + "'" + ")"
            
            
    sql_como += " group by personid"
    
    print(sql_como)        
    como_processed = spark.sql(sql_como)
    como_processed.cache()
    
    return como_processed
    

def process_lab(spark, cohort, lab):
    cohort.createOrReplaceTempView('cohort_temp')
    lab.createOrReplaceTempView('lab_temp')
    
    lab_processed = spark.sql("""
        with temp1 as (
        select distinct r.personid, r.date, l.encounterid, min(l.servicedate) as earliest_service
        from
        lab_temp l
        inner join
        cohort_temp r
        on l.personid = r.personid
        group by  r.personid
        )
        select l.personid, r.labcode, r.value, r.servicedate
        from 
        temp1 l 
        inner join
        lab_temp r
        on l.personid = r.personid
        where l.earliest_service = r.servicedate
    """)
    
    lab_processed = lab_processed.groupBy('personid').pivot('labcode').agg(avg('value'))
    lab_processed.cache()
    
    return lab_processed

def process_lab_spark(spark, cohort, lab, start=1, end=0):
#     e.g. start = 24, end = 6: 6-24 months before diagnosis
    joined = lab.alias('l').select('personid','labcode','servicedate','value').join(cohort.alias('c'), 'personid')
    joined_filtered = joined.filter((months_between(col('date'),col('servicedate')) >= end) & (months_between(col('date'),col('servicedate')) <= start))
    joined_filtered_small = joined_filtered.\
    select('personid','labcode','value','servicedate').distinct().\
    groupBy('personid','labcode','servicedate').agg(avg('value').alias('avg_value'))\
    .select('personid','labcode','avg_value').groupBy('personid','labcode').agg(avg('avg_value').alias('avg_v'))
    joined_filtered_pivot = joined_filtered_small.groupBy('personid').pivot('labcode').agg(avg('avg_v'))
    return joined_filtered_pivot

def process_control_lab(spark, cohort, lab):
    cohort.createOrReplaceTempView('cohort_temp')
    lab.createOrReplaceTempView('lab_temp')
    
    lab_processed = lab.select('personid','labcode','value','servicedate')
    
    lab_processed = lab_processed.groupBy('personid').pivot('labcode').agg(avg('value'))
    lab_processed.cache()
    
    return lab_processed


def union_demo_como_lab(spark, demo, como, lab,outputfilename="", outputfolder = ""):
    if outputfilename =="":
        print("please set outputfilename!")
        return None
    
    if outputfolder =="":
        print("please set outputfolder!")
        return None
    df = demo.join(como,['personid']).join(lab,['personid'])
    df.write.mode('overwrite').parquet(outputfilename)
    df.cache()
    os.system("hadoop fs -copyToLocal " + outputfilename + " ~/work/Oklahoma%20State/" + outputfolder)
    return df

def extract_control_pt_from_condition(spark, icdlist=[], icdlikelist=[],outputfilename="", outputfolder = ""):
#     items in icdlist and icdlikes should be single quote mark
# the outputfolder should be like "Zheng_Han/pompe". It is appended within the Oklahoma state folder. No relative path is allowed.

    if outputfilename =="":
        print("please set outputfilename!")
        return None
    
    if outputfolder =="":
        print("please set outputfolder!")
        return None
        
    sql_sentence = "select personid, encounterid, effectivedate as date, conditioncode.standard.id as conditioncode from condition_s where "
    cnt = 0
    insentence = ""
    likesentence = ""
    
    if len(icdlist) > 0:
        insentence = "conditioncode.standard.id not in ("
        for i in range(len(icdlist)-1):
            insentence = insentence + "'" + icdlist[i] + "'" + ","
        insentence = insentence + "'" +icdlist[-1] + "'"
        insentence += ")"
        
        cnt +=1
    
    if len(icdlikelist) > 0:
        likesentence = "conditioncode.standard.id not like "
        for i in range(len(icdlikelist)-1):
            likesentence = likesentence + "'" + icdlikelist[i] + "'" 
            likesentence += " and conditioncode.standard.id not like "

        likesentence = likesentence + "'" + icdlikelist[-1] +  "'" 
        cnt +=1
        
    if cnt == 2:
        sql_sentence = sql_sentence + insentence + " and " + likesentence
    else:
        sql_sentence = sql_sentence + insentence + likesentence
          
    print(sql_sentence)
    df = spark.sql(sql_sentence)
    df.write.mode('overwrite').parquet(outputfilename)
    df.cache()
    os.system("hadoop fs -copyToLocal " + outputfilename + " ~/work/Oklahoma%20State/" + outputfolder)
    return df

def cohort_distribution(spark, cohort_demo):
    cohort_demo = cohort_demo.distinct()
    cohort_demo.cache()
    genderCnt = cohort_demo.select('personid','gender').na.drop().distinct().groupBy('gender').count().orderBy(col('count').desc())
    genderCnt.show()
    
    raceCnt = cohort_demo.select('personid','race').na.drop().distinct().groupBy('race').count().orderBy(col('count').desc())
    raceCnt.show()
    
    cohort_demo = cohort_demo.withColumn('age',lit(months_between(current_date(), col('birthdate'))/12)).select('personid','age').na.drop().distinct()
    print('age<18',cohort_demo.filter(cohort_demo.age <18).count())
    print('18<=age<44',cohort_demo.filter((cohort_demo.age <44) & (cohort_demo.age >=18)).count())
    print('44<=age<60',cohort_demo.filter((cohort_demo.age <60) & (cohort_demo.age >=44)).count())
    print('age>=60',cohort_demo.filter(cohort_demo.age >=60).count())
    
def extractControlComorbidity(spark, cohort, top_como_dict = "", outputfilename="", outputfolder = ""):
#     items in icdlist and icdlikes should be single quote mark
# the outputfolder should be like "Zheng_Han/pompe". It is appended within the Oklahoma state folder. No relative path is allowed.

    if outputfilename =="":
        print("please set outputfilename!")
        return None
    
    if outputfolder =="":
        print("please set outputfolder!")
        return None
    
    cohort.createOrReplaceTempView('cohort')
    
    insentence = ""
        
    sql_sentence = "select distinct l.personid, l.encounterid, l.conditioncode.standard.id as comorbidityid, "
    sql_sentence += "l.effectivedate,l.classification.standard.id as classification, l.confirmationstatus.standard.primaryDisplay "
    sql_sentence += "from condition l inner join cohort r on l.personid = r.personid where "
    
    
    if len(top_como_dict) > 0:
        insentence = "l.conditioncode.standard.id in ("
        for i,(k,v) in enumerate(top_como_dict.items()):
            if i < len(top_como_dict) - 1:
                insentence = insentence + "'" + k + "'" + ","
            else:
                insentence = insentence + "'" + k + "')"
    
    
    sql_sentence = sql_sentence + insentence
          
          
    print(sql_sentence)
    df = spark.sql(sql_sentence)
    df.write.mode('overwrite').parquet(outputfilename)
    os.system("hadoop fs -copyToLocal " + outputfilename + " ~/work/Oklahoma%20State/" + outputfolder)
    return df

from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StandardScaler,OneHotEncoder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer

def flatlist(l):
    out = [item for sublist in l for item in sublist]
    return out
def expandVector(vector,targetColumn,expandColumns):
    rdd=vector.rdd.map(lambda x: 
    (
         flatlist([x,[float(i) for i in x[targetColumn]]])
    ))
    out = rdd.toDF(vector.columns + expandColumns)
    return out


def ohe_gender_race(df):
    indexer = StringIndexer(inputCol='gender', outputCol='gender_numeric')
    indexer_fitted = indexer.fit(df)
    df_indexed = indexer_fitted.transform(df)
    # df_indexed.show()

    print(indexer_fitted.labels)

    ohe = OneHotEncoder(inputCol='gender_numeric', outputCol='gender_onehot',dropLast=False)
    # model = ohe.fit(df_indexed)
    df_onehot = ohe.transform(df_indexed)
#     print(df_onehot.columns)
    df_onehot = expandVector(df_onehot,'gender_onehot',indexer_fitted.labels)
    
    indexer = StringIndexer(inputCol='race', outputCol='race_numeric')
    indexer_fitted = indexer.fit(df_onehot )
    df_indexed = indexer_fitted.transform(df_onehot )
    # df_indexed.show()

    print(indexer_fitted.labels)

    ohe = OneHotEncoder(inputCol='race_numeric', outputCol='race_onehot',dropLast=False)
    # model = ohe.fit(df_indexed)
    df_onehot = ohe.transform(df_indexed)
    df_onehot = expandVector(df_onehot,'race_onehot',indexer_fitted.labels)
    
    df_onehot = df_onehot.drop('gender','gender_numeric','race','race_numeric')
    
    return df_onehot
    
    

    