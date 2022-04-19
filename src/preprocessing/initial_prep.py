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
    Crashes=Crashes[Crashes['REPORT_TYPE']=='ON SCENE']
    Crashes=Crashes.query("CRASH_TYPE == 'INJURY AND / OR TOW DUE TO CRASH'")


    latest=date.today().year
    oldest=Vehicle['VEHICLE_YEAR'].min()

    Vehicle=Vehicle.copy()
    vehicle_age=[oldest-1, latest-25, latest-15, latest-8, latest-3, latest ]
    vehicle_age_labels=['25+ years old', '15-25 years old','8-15 years old','3-8 years old','0-3 years old']
    Vehicle['VEHICLE_AGE'] = pd.cut(x=Vehicle['VEHICLE_YEAR'], bins=vehicle_age, labels=vehicle_age_labels, right=False)


    Vehicle.dropna(subset=['FIRST_CONTACT_POINT'], inplace=True)
    veh_type=['PASSENGER','SPORT UTILITY VEHICLE (SUV)','VAN/MINI-VAN','PICKUP','TRUCK - SINGLE UNIT','BUS OVER 15 PASS.']
    Vehicle=Vehicle[Vehicle['VEHICLE_TYPE'].isin(veh_type)]

    People = People.query('PERSON_TYPE == "DRIVER"')
    People = People.query('(AGE > 14 and AGE < 101) | AGE.isnull()', engine='python')  #
    People = People.query('DRIVERS_LICENSE_CLASS in ("A","B","C","D","DM","AM","BM","CD") | DRIVERS_LICENSE_CLASS.isnull()', engine='python')
    People=People.query('SEX!="X"')

    People=People.copy()
    People['AGE_GROUP'] = pd.cut(x=People['AGE'], bins=[15, 24, 35, 55, 70, 80, 100],
                             labels=['YOUNG', 'YOUNG ADULTS','ADULTS','MIDDLE AGED','SENIORS', '80+'])

    temp_df1=pd.merge(People, Vehicle, on=['CRASH_RECORD_ID','VEHICLE_ID'])
    all_df=pd.merge(temp_df1, Crashes, on='CRASH_RECORD_ID')
    return all_df


def create_target(this_driver_action, driver_error, driver_sec_error, phys_condition):
    if this_driver_action not in ['NONE','OTHER','UNKNOWN']:
        guilty='YES'
    elif this_driver_action == 'NONE':
        guilty='NO'
    elif this_driver_action in ['OTHER','UNKNOWN'] and phys_condition in ['MEDICATED','FATIGUED/ASLEEP','ILLNESS/FAINTED'] :
        guilty='NO'        
    else: 
        guilty = driver_error

    if guilty == 'UNKNOWN' and driver_error == 'UNKNOWN' and driver_sec_error == 'NO':
        guilty='NO'
    return guilty


People_df=pd.read_csv("../data/raw/Traffic_Crashes_-_People.csv.gz", low_memory=False, compression='gzip')
Crashes_df=pd.read_csv("../data/raw/Traffic_Crashes_-_Crashes.csv.gz",low_memory=False, compression='gzip')
Vehicle_df=pd.read_csv("../data/raw/Traffic_Crashes_-_Vehicles.csv.gz",low_memory=False, compression='gzip')


# Creating the dataframe
all_df=initial_join( Crashes_df, Vehicle_df, People_df) 


# GENERATING TARGET THROUGH binning and combining sec and prim causes:

targ_map ={ 'IMPROPER OVERTAKING/PASSING':'YES', 'UNABLE TO DETERMINE':'UNKNOWN',
       'IMPROPER BACKING':'YES', 'IMPROPER LANE USAGE':'YES',
       'UNDER THE INFLUENCE OF ALCOHOL/DRUGS (USE WHEN ARREST IS EFFECTED)':'YES',
       'DISREGARDING TRAFFIC SIGNALS':'YES',
       'FAILING TO REDUCE SPEED TO AVOID CRASH':'YES',
       'OPERATING VEHICLE IN ERRATIC, RECKLESS, CARELESS, NEGLIGENT OR AGGRESSIVE MANNER':'YES',
       'FAILING TO YIELD RIGHT-OF-WAY':'YES', 'EQUIPMENT - VEHICLE CONDITION':'NO',
       'VISION OBSCURED (SIGNS, TREE LIMBS, BUILDINGS, ETC.)':'NO',
       'IMPROPER TURNING/NO SIGNAL':'YES', 'FOLLOWING TOO CLOSELY':'YES',
       'DRIVING SKILLS/KNOWLEDGE/EXPERIENCE':'YES', 'DISREGARDING STOP SIGN':'YES',
       'NOT APPLICABLE':'UNKNOWN', 'DISTRACTION - FROM INSIDE VEHICLE':'NO',
       'DISTRACTION - FROM OUTSIDE VEHICLE':'NO',
       'HAD BEEN DRINKING (USE WHEN ARREST IS NOT MADE)':'YES',
       'ROAD ENGINEERING/SURFACE/MARKING DEFECTS':'NO',
       'DISREGARDING OTHER TRAFFIC SIGNS':'YES', 'TEXTING':'YES',
       'DRIVING ON WRONG SIDE/WRONG WAY':'YES', 'PHYSICAL CONDITION OF DRIVER':'YES',
       'ANIMAL':'NO', 'WEATHER':'NO', 'ROAD CONSTRUCTION/MAINTENANCE':'NO',
       'DISREGARDING YIELD SIGN':'YES', 'CELL PHONE USE OTHER THAN TEXTING':'YES',
       'EVASIVE ACTION DUE TO ANIMAL, OBJECT, NONMOTORIST':'NO',
       'TURNING RIGHT ON RED':'YES', 'RELATED TO BUS STOP':'NO',
       'DISTRACTION - OTHER ELECTRONIC DEVICE (NAVIGATION DEVICE, DVD PLAYER, ETC.)':'YES',
       'DISREGARDING ROAD MARKINGS':'YES', 'OBSTRUCTED CROSSWALKS':'NO',
       'PASSING STOPPED SCHOOL BUS':'YES',
       'EXCEEDING SAFE SPEED FOR CONDITIONS':'YES',
       'EXCEEDING AUTHORIZED SPEED LIMIT':'YES',
       'MOTORCYCLE ADVANCING LEGALLY ON RED LIGHT':'NO',
       'BICYCLE ADVANCING LEGALLY ON RED LIGHT':'NO'}

all_df['SOME_DRIVER_ERROR']=all_df['PRIM_CONTRIBUTORY_CAUSE'].replace(targ_map)
all_df['SOME_DRIVER_SEC_ERROR']=all_df['SEC_CONTRIBUTORY_CAUSE'].replace(targ_map)

all_df['GUILTY']=all_df.apply(lambda row: create_target(row['DRIVER_ACTION'], row['SOME_DRIVER_ERROR'], 
                                                        row['SOME_DRIVER_SEC_ERROR'],row['PHYSICAL_CONDITION']), axis=1)
# DROPPING UNKNOWN TARGET
all_df = all_df[all_df['GUILTY']!='UNKNOWN']

y=all_df['GUILTY']
all_col=all_df.drop(['GUILTY'],axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(all_col, y, random_state=100, stratify=y)

train_df=pd.concat([X_train,y_train], axis=1)
test_df=pd.concat([X_test,y_test], axis=1)




