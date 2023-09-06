import shutil
import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql.functions import *

    
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.5)

def showdatabases(spark):
    spark.sql("SHOW databases").show()
    
def showtables(spark):
    spark.sql("SHOW tables").show()
    
def value_counts(df, colName):
    return (df.groupby(colName).count()
              .orderBy('count', ascending=False))

def usedatabase(spark, databasename):
    spark.sql("USE "+ databasename)
    print("Using "+databasename+" ....")
    
def gettable(spark, tablename):
    return spark.table(tablename)
    
def jointable(spark, table1, table2, joinkey):
    out = table1.join(table2,joinkey,'inner')        
    return out

def select(spark, table, columns, alias):
    out = []
    return out

def delete_parquet(path):
    shutil.rmtree(path)
    
# def save_parquet(df,path, name):
#     filename = name+.'parquet'
#     curdir = os.getcwd()
#     outpath = os.path.join(curdir,path)
#     df.write.parquet(filename)
#     !hadoop fs -copyToLocal filename ~/work/Oklahoma%20State/Zheng_Han

def visualize_topx(df, key, x):
    arr = df[key].value_counts()[0:x]
    print(arr)
    plt.barh(arr.index,arr.values)
    
def convertBirthdateToAge(df, birthdatecolumn, newcolumnname, remove=True):
    # example: dementia_demo_converted = convertBirthdateToAge(dementia_demo_pd,'birthdate','age')
    today = datetime.date.today()
    # df['dob'] = df['dob'].apply('{:06}'.format)

    now = pd.Timestamp('now')
    df[birthdatecolumn] = pd.to_datetime(df[birthdatecolumn], format='%Y-%m-%dT%H:%M:%S.%f', errors = 'coerce')    # 1
    df[birthdatecolumn] = df[birthdatecolumn].where(df[birthdatecolumn] < now, df[birthdatecolumn] -  np.timedelta64(100, 'Y'))   # 2
    df[newcolumnname] = (now - df[birthdatecolumn]).astype('<m8[Y]')    # 3
    df[newcolumnname] = df[newcolumnname].astype('Int64')
    df = df.drop(columns =[birthdatecolumn])
    return df

def cntAdult_Child(df_path):
#     example: _,_ = cntAdult_Child('rsv_birthdate.parquet')
    df_pq = pq.read_table(df_path)
    df_pd = df_pq.to_pandas()
    df_age_pd = util.convertBirthdateToAge(df_pd,'birthdate','age').dropna().drop_duplicates(ignore_index=True)
    df_young = df_age_pd[df_age_pd.age < 18]
    df_old = df_age_pd[df_age_pd.age >= 18]
    print("The number of children (<18 years) is: ", df_young.personid.nunique())
    print("The number of adults (>=18 years) is: ", df_old.personid.nunique())
    return df_young, df_old

def xgbPlotFeatureImportance(model_xgb):
    df = pd.DataFrame({'features':model_xgb.get_booster().feature_names, 'importance':model_xgb.feature_importances_}).convert_dtypes()
    df = df.sort_values(by='importance')
    print(df)
    plt.figure(figsize=(10,10))
    plt.barh(df.features, df.importance)
    plt.xlabel('features')
    plt.ylabel('importance')
    
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
    
def cohort_distribution_age(spark, cohort_demo):
    cohort_demo = cohort_demo.distinct()    
    cohort_demo = cohort_demo.withColumn('age',lit(months_between(current_date(), col('birthdate'))/12)).select('personid','age').na.drop().distinct()
    print('age<18',cohort_demo.filter(cohort_demo.age <18).count())
    print('18<=age<44',cohort_demo.filter((cohort_demo.age <44) & (cohort_demo.age >=18)).count())
    print('44<=age<60',cohort_demo.filter((cohort_demo.age <60) & (cohort_demo.age >=44)).count())
    print('age>=60',cohort_demo.filter(cohort_demo.age >=60).count())
    
    
def cntPtAgeGroup(df_pq, ages=[18,44,65]):
#     df_pq = pq.read_table(df_path)
    df_pd = df_pq.to_pandas()
    df_age_pd = util.convertBirthdateToAge(df_pd,'birthdate','age')[['personid','age']].dropna().drop_duplicates(ignore_index=True)
    print("total number of patients with age is: ",df_age_pd.personid.nunique())
    df = df_age_pd[df_age_pd.age < ages[0]]
    print("The number of children (<",ages[0]," years) is: ", df.personid.nunique())
    for i in range(len(ages)-1):
        df= df_age_pd[(df_age_pd.age >= ages[i]) & (df_age_pd.age < ages[i+1])]
        print("The number of adults (>=", ages[i]," and <",ages[i+1],"years) is: ", df.personid.nunique())
    
    df = df_age_pd[df_age_pd.age > ages[len(ages)-1]]
    print("The number of adults (>=",ages[len(ages)-1], " years) is: ", df.personid.nunique())
    

def plot_yearMonth(spark, sparkdf, colname, datename,x_tick_count=20):
    sparkdf.createOrReplaceTempView('temp_df')
    sqlsentence = "select count(distinct "+ colname + ") as no_cases, substring(" + datename + ",1,7) as year_month \
    from temp_df where startdate <> ''\
    group by year_month \
    order by year_month"
    print(sqlsentence)
    df = spark.sql(sqlsentence)
    data = df.toPandas()

    no_cases = data.no_cases
    date = data.year_month

    fig, ax = plt.subplots(figsize=(20, 10))
    g = sns.lineplot(date,no_cases)

    dilute_factor = np.round(len(data)/x_tick_count)

    for index, label in enumerate(g.get_xticklabels()):
       if index % dilute_factor == 0:
          label.set_visible(True)
       else:
          label.set_visible(False)
    
    plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees

    plt.show()
    return df
    
