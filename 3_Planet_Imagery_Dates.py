#%%Modules
from IPython import get_ipython
get_ipython().magic('reset -sf')

import os
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
#%%
Loc = "LD1"
Type = 'Stats' #Stats or Eqint

fp = r"C:\\Users\\OCRONING\\OneDrive - Environmental Protection Agency (EPA)\\Profile\\Documents\\PFAS_Foam\\"
In = fp + r"Inputs\\"
Out = fp + "Outputs\\" + Loc 

Discharge_all = pd.read_csv(In + "Discharge\\Discharge_" + Loc + ".csv")
Discharge = pd.read_csv(In + "Target_imagery\\" + Loc + "\\Discharge_" + Type + "_Representative_Imagery_" + Loc + ".csv")

#%%
Start_year = str(Discharge_all.year.min())
Start_month = str(Discharge_all.month.min()).zfill(2)

End_year = str(Discharge_all.year.max())
End_month = str(Discharge_all.month.max()).zfill(2)

#%%Planet API Key
if os.environ.get('PL_API_KEY', ''):
    API_KEY = os.environ.get('PL_API_KEY', '')
else:
    API_KEY = 'PLAKd93a8f9d424243788e84ac8210d4e37c'
    
#%% Target AOI of planet imagery
AOI_geometry = pd.read_csv(In + "Shapefiles\\" + Loc + "\\" + Loc + "_AOI_pts.csv")
AOI_geometry = AOI_geometry.drop(['OID_'], axis = 1)
AOI_coords = AOI_geometry.values.tolist()

geojson_geometry = {
  "type": "Polygon",
  "coordinates": [
     AOI_coords
  ]
}

#%%
# get images that overlap with our AOI 
geometry_filter = {
  "type": "GeometryFilter",
  "field_name": "geometry",
  "config": geojson_geometry
}

# get images acquired within a date range
date_range_filter = {
  "type": "DateRangeFilter",
  "field_name": "acquired",
  "config": {
    "gte": Start_year + "-" + Start_month + "-01T00:00:00.000Z",
    "lte": "2024" + "-" + "01" + "-31T00:00:00.000Z"
  }
}

# only get images which have <10% cloud coverage
cloud_cover_filter = {
  "type": "RangeFilter",
  "field_name": "cloud_cover",
  "config": {
    "lte": 0.1
  }
}

# combine our geo, date, cloud filters
combined_filter = {
  "type": "AndFilter",
  "config": [geometry_filter, date_range_filter, cloud_cover_filter]
}

#%%
item_type = "PSScene"

# API request object
search_request = {
  "item_types": [item_type], 
  "filter": combined_filter
}

# fire off the POST request
search_result = \
  requests.post(
    'https://api.planet.com/data/v1/quick-search',
    auth=HTTPBasicAuth(API_KEY, ''),
    json=search_request)

geojson = search_result.json()

# extract image IDs only
image_ids = [feature['id'] for feature in geojson['features']]

#%% Get dates of planet imagery
Year = []    
for i in range(len(image_ids)):
    Year.append(int(image_ids[i][0:4:1]))
    
Month = []    
for i in range(len(image_ids)):
    Month.append(int(image_ids[i][4:6:1]))
    
Day = []    
for i in range(len(image_ids)):
    Day.append(int(image_ids[i][6:8:1]))
    
#%% Select planet imagery taken on days with the target discharge 
Planet_imgs = pd.DataFrame(list(zip(Year, Month, Day, image_ids)), columns=['year', 'month', 'day', 'planet_id'])

Matches = pd.merge(Planet_imgs, Discharge,  how='right', left_on=['year','month','day'], right_on = ['year','month','day'])
Matches = Matches.dropna(subset = ['planet_id']) 

Qualifier = Matches.Qualifier.unique().tolist()

Planet_target = []
for i in range(len(Qualifier)):
    df = Matches[Matches.Qualifier == Qualifier[i]]
    df1 = df[df.year == df.year.max()]
    df2 = df1[df1.month == df1.month.max()]
    df3 = df2[df2.day == df2.day.max()]
    
    Planet_target.append(df3)

Planet_target = pd.concat(Planet_target)
newdf1 = Planet_target.groupby(['month', 'day']).first() 
newdf1 = newdf1.reset_index()

Exact_discharge = newdf1.loc[:,['year','month','day', 'planet_id', 'Discharge_cfps', 'Qualifier']]
Exact_discharge['Diff_cfps'] = '0'
Exact_discharge['Target_Discharge'] = Exact_discharge['Discharge_cfps']

Exact_discharge = Exact_discharge.loc[:,['year','month','day', 'planet_id', 'Qualifier', 'Target_Discharge', 'Discharge_cfps', 'Diff_cfps']]
Exact_discharge = Exact_discharge.rename(columns={'Discharge_cfps': 'Actual_discharge'})

del(df, df1, df2, df3)

#%% Subset values that dont have an exact discharge imagery match
Quant_nomatch = Discharge.Qualifier.unique()
Quant_nomatch.sort()

noMatch = list(set(Quant_nomatch) - set(Qualifier))
NoMatch = Discharge[Discharge.Qualifier.isin(noMatch)]

noMatchTarget = []
for i in range(len(noMatch)):
    df = NoMatch[NoMatch.Qualifier == noMatch[i]]
    df1 = df[df.year == df.year.max()]
    df2 = df1[df1.month == df1.month.max()]
    df3 = df2[df2.day == df2.day.max()]
    noMatchTarget.append(df3)
noMatchTarget = pd.concat(noMatchTarget)

newdf2 = noMatchTarget.groupby(['month', 'day']).first() 
newdf2 = newdf2.reset_index()
noMatchTarget = newdf2.sort_values('Discharge_cfps')

del(df, df1, df2, df3, newdf2)
#%% Get discharge values close to the target discharge
Discharge_all = Discharge_all[['year','month','day', 'Discharge_cfps']]
Discharge_all = Discharge_all.dropna(subset = ['Discharge_cfps'])
Discharge_all = Discharge_all.sort_values('Discharge_cfps')

test = pd.merge_asof(Discharge_all, noMatchTarget, on='Discharge_cfps', direction='nearest', tolerance = 100, allow_exact_matches = False)
test1 = test.dropna() 
test2 = test1[['year_x','month_x','day_x', 'Discharge_cfps','Qualifier']]
test2 = test2.rename(columns={'year_x': 'year', 'month_x':'month', 'day_x':'day'})

Closer = pd.merge(Planet_imgs, test2,  how='right', left_on=['year','month','day'], right_on = ['year','month','day']).dropna().drop_duplicates()

#%% Get discharge value closest to target discharge
Closest_discharge = []

for i in range(len(noMatch)):
    df = NoMatch[NoMatch.Qualifier == noMatch[i]]
    targDischarge = df['Discharge_cfps'].iloc[0]
    nearDischarge = Closer[Closer.Qualifier == noMatch[i]]
    nearDischarge['Target_Discharge'] = targDischarge   
    nearDischarge['Diff_cfps'] = nearDischarge['Discharge_cfps'] - nearDischarge['Target_Discharge']
    Closest = nearDischarge.iloc[(nearDischarge['Diff_cfps']-0).abs().argsort()[:1]]
    Closest_discharge.append(Closest)
Closest_discharge = pd.concat(Closest_discharge)

Closest_discharge = Closest_discharge.loc[:,['year','month','day', 'planet_id', 'Qualifier', 'Target_Discharge', 'Discharge_cfps', 'Diff_cfps']]
Closest_discharge = Closest_discharge.rename(columns={'Discharge_cfps': 'Actual_discharge'})

#%%
Discharge_matchups = pd.concat([Exact_discharge, Closest_discharge])
Discharge_matchups = Discharge_matchups.sort_values('Qualifier')
Discharge_matchups.to_csv(In + "\\Target_imagery\\" + Loc + "\\Planet_Target_Imagery_" + Type + "_" + Loc + ".csv", index=False)