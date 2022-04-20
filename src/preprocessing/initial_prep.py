#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, date
from pysolar.solar import *
from zoneinfo import ZoneInfo
from dateutil import tz

pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 200)

def initial_join (Crashes, Vehicle, People):

    # INITIAL FILTERING BEFORE the JOIN, FOR PERFORMANCE REASONS:
#    Crashes=Crashes[Crashes['REPORT_TYPE']=='ON SCENE']
#    Crashes=Crashes.query("CRASH_TYPE == 'INJURY AND / OR TOW DUE TO CRASH'")

    latest=date.today().year
    oldest=Vehicle['VEHICLE_YEAR'].min()

    Vehicle=Vehicle[Vehicle['VEHICLE_YEAR']<=date.today().year]
    Vehicle=Vehicle[Vehicle['VEHICLE_YEAR']>1900]

    Vehicle=Vehicle.copy()
    vehicle_age=[oldest-1, latest-25, latest-15, latest-8, latest-3, latest ]
    vehicle_age_labels=['25+ years old', '15-25 years old','8-15 years old','3-8 years old','0-3 years old']
    Vehicle['VEHICLE_AGE'] = pd.cut(x=Vehicle['VEHICLE_YEAR'], bins=vehicle_age, labels=vehicle_age_labels, right=False)


    Vehicle.dropna(subset=['FIRST_CONTACT_POINT'], inplace=True)
    veh_type=['PASSENGER','SPORT UTILITY VEHICLE (SUV)','VAN/MINI-VAN','PICKUP','TRUCK - SINGLE UNIT','BUS OVER 15 PASS.']
    Vehicle=Vehicle[Vehicle['VEHICLE_TYPE'].isin(veh_type)]

    People = People.query('PERSON_TYPE == "DRIVER"')
#   People = People.query('(AGE > 14 and AGE < 101) | AGE.isnull()', engine='python')  
    People = People.query('(AGE > 14 and AGE < 101)', engine='python')  # will remove na
#    People = People.query('DRIVERS_LICENSE_CLASS in ("A","B","C","D","DM","AM","BM","CD") | DRIVERS_LICENSE_CLASS.isnull()', engine='python')
    People = People.query('DRIVERS_LICENSE_CLASS in ("A","B","C","D","DM","AM","BM","CD")', engine='python') # will remove na
    People=People.query('SEX!="X"')

    People=People.copy()
    People['AGE_GROUP'] = pd.cut(x=People['AGE'], bins=[15, 24, 35, 55, 70, 80, 100],
                             labels=['YOUNG', 'YOUNG ADULTS','ADULTS','MIDDLE AGED','SENIORS', '80+'])

    temp_df1=pd.merge(People, Vehicle, on=['CRASH_RECORD_ID','VEHICLE_ID'])
    all_df=pd.merge(temp_df1, Crashes, on='CRASH_RECORD_ID')
    return all_df

def drop_unknown (df):
    new_df=df.copy()
    info=[]
    total = df.shape[0]
    for col in df.columns:
        unknown_n=round((df[col]=='UNKNOWN').sum()/total*100,2)
        not_na = round(df[col].isna().sum()/total*100,2)
        na_and_unknown=unknown_n+not_na

        if unknown_n > 0 and unknown_n < 50 :
#        if unknown_n > 0 and na_and_unknown < 50 :
            new_df=df[df[col]!='UNKNOWN']
    return new_df

def create_target(this_driver_action, driver_error, driver_sec_error, phys_condition):
    if this_driver_action not in ['NONE','OTHER','UNKNOWN']:
        guilty=1
    elif this_driver_action == 'NONE':
        guilty=0
    elif this_driver_action in ['OTHER','UNKNOWN'] and phys_condition in ['MEDICATED','FATIGUED/ASLEEP','ILLNESS/FAINTED'] :
        guilty=0        
    else: 
        guilty = driver_error

    if guilty == 'UNKNOWN' and driver_error == 'UNKNOWN' and driver_sec_error == 0:
        guilty=0
    return guilty


People_df=pd.read_csv("../data/raw/Traffic_Crashes_-_People.csv.gz", low_memory=False, compression='gzip')
Crashes_df=pd.read_csv("../data/raw/Traffic_Crashes_-_Crashes.csv.gz",low_memory=False, compression='gzip')
Vehicle_df=pd.read_csv("../data/raw/Traffic_Crashes_-_Vehicles.csv.gz",low_memory=False, compression='gzip')


# Creating the dataframe
all_df=initial_join( Crashes_df, Vehicle_df, People_df) 


# GENERATING TARGET THROUGH binning and combining sec and prim causes:

targ_map ={ 'IMPROPER OVERTAKING/PASSING':1, 'UNABLE TO DETERMINE':'UNKNOWN',
       'IMPROPER BACKING':1, 'IMPROPER LANE USAGE':1,
       'UNDER THE INFLUENCE OF ALCOHOL/DRUGS (USE WHEN ARREST IS EFFECTED)':1,
       'DISREGARDING TRAFFIC SIGNALS':1,
       'FAILING TO REDUCE SPEED TO AVOID CRASH':1,
       'OPERATING VEHICLE IN ERRATIC, RECKLESS, CARELESS, NEGLIGENT OR AGGRESSIVE MANNER':1,
       'FAILING TO YIELD RIGHT-OF-WAY':1, 'EQUIPMENT - VEHICLE CONDITION':0,
       'VISION OBSCURED (SIGNS, TREE LIMBS, BUILDINGS, ETC.)':0,
       'IMPROPER TURNING/NO SIGNAL':1, 'FOLLOWING TOO CLOSELY':1,
       'DRIVING SKILLS/KNOWLEDGE/EXPERIENCE':1, 'DISREGARDING STOP SIGN':1,
       'NOT APPLICABLE':'UNKNOWN', 'DISTRACTION - FROM INSIDE VEHICLE':0,
       'DISTRACTION - FROM OUTSIDE VEHICLE':0,
       'HAD BEEN DRINKING (USE WHEN ARREST IS NOT MADE)':1,
       'ROAD ENGINEERING/SURFACE/MARKING DEFECTS':0,
       'DISREGARDING OTHER TRAFFIC SIGNS':1, 'TEXTING':1,
       'DRIVING ON WRONG SIDE/WRONG WAY':1, 'PHYSICAL CONDITION OF DRIVER':1,
       'ANIMAL':0, 'WEATHER':0, 'ROAD CONSTRUCTION/MAINTENANCE':0,
       'DISREGARDING YIELD SIGN':1, 'CELL PHONE USE OTHER THAN TEXTING':1,
       'EVASIVE ACTION DUE TO ANIMAL, OBJECT, NONMOTORIST':0,
       'TURNING RIGHT ON RED':1, 'RELATED TO BUS STOP':0,
       'DISTRACTION - OTHER ELECTRONIC DEVICE (NAVIGATION DEVICE, DVD PLAYER, ETC.)':1,
       'DISREGARDING ROAD MARKINGS':1, 'OBSTRUCTED CROSSWALKS':0,
       'PASSING STOPPED SCHOOL BUS':1,
       'EXCEEDING SAFE SPEED FOR CONDITIONS':1,
       'EXCEEDING AUTHORIZED SPEED LIMIT':1,
       'MOTORCYCLE ADVANCING LEGALLY ON RED LIGHT':0,
       'BICYCLE ADVANCING LEGALLY ON RED LIGHT':0}

all_df['SOME_DRIVER_ERROR']=all_df['PRIM_CONTRIBUTORY_CAUSE'].replace(targ_map)
all_df['SOME_DRIVER_SEC_ERROR']=all_df['SEC_CONTRIBUTORY_CAUSE'].replace(targ_map)

all_df['GUILTY']=all_df.apply(lambda row: create_target(row['DRIVER_ACTION'], row['SOME_DRIVER_ERROR'], 
                                                        row['SOME_DRIVER_SEC_ERROR'],row['PHYSICAL_CONDITION']), axis=1)
# DROPPING UNKNOWN TARGET

#cleaned_df=drop_unknown(all_df)

all_df = all_df[all_df['GUILTY']!='UNKNOWN']

all_df=all_df[all_df['PHYSICAL_CONDITION'] != 'UNKNOWN']
all_df=all_df[all_df['VEHICLE_DEFECT'] != 'UNKNOWN']
all_df=all_df[all_df['MANEUVER'] != 'UNKNOWN/NA']
all_df=all_df[all_df['FIRST_CONTACT_POINT'] != 'UNKNOWN']
all_df=all_df[all_df['TRAFFIC_CONTROL_DEVICE'] != 'UNKNOWN']
all_df=all_df[all_df['DEVICE_CONDITION'] != 'UNKNOWN']
all_df=all_df[all_df['WEATHER_CONDITION'] != 'UNKNOWN']
all_df=all_df[all_df['LIGHTING_CONDITION'] != 'UNKNOWN']
all_df=all_df[all_df['TRAFFICWAY_TYPE'] !=  'UNKNOWN']
all_df=all_df[all_df['ROADWAY_SURFACE_COND']!= 'UNKNOWN']
all_df=all_df[all_df['ROAD_DEFECT'] != 'UNKNOWN']
all_df=all_df[all_df['DRIVER_VISION'] != 'UNKNOWN']
all_df=all_df[all_df['AIRBAG_DEPLOYED'] != 'DEPLOYMENT UNKNOWN']

y=all_df['GUILTY']
all_col=all_df.drop(['GUILTY'],axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(all_col, y, random_state=100, stratify=y)

train_df=pd.concat([X_train,y_train], axis=1)
test_df=pd.concat([X_test,y_test], axis=1)




