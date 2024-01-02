#%%Modules
from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
from collections import Counter 
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import math

#%%Parameters
Loc = "LD1"
fp = r"C:\\Users\\OCRONING\\OneDrive - Environmental Protection Agency (EPA)\\Profile\\Documents\\PFAS_Foam\\"
In = fp + r"Inputs\\"
Out = fp + "Outputs\\" + Loc + "\\"

Discharge = pd.read_csv(In + "Discharge\\Discharge_" + Loc + ".csv")
Discharge = Discharge.dropna(subset = ['Discharge_cfps'])

#%% Plot Discharge Histogram
plt.hist(Discharge['Discharge_cfps'], bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Discharge [cfps]')
plt.ylabel('Frequency')
plt.title('Discharge Distribution of ' + Loc)
plt.show()

#%% Use stats to figure out which images to download from Planet
Discharge_min = Discharge['Discharge_cfps'].min()
Discharge_max = Discharge['Discharge_cfps'].max()
Discharge_median = statistics.median_low(Discharge['Discharge_cfps'])

Discharge_25 = np.percentile(Discharge['Discharge_cfps'], 25, method = 'nearest')
Discharge_75 = np.percentile(Discharge['Discharge_cfps'], 75, method = 'nearest')

n = len( Discharge['Discharge_cfps']) 
data = Counter(Discharge['Discharge_cfps']) 
get_mode = dict(data) 
Discharge_mode = [k for k, v in get_mode.items() if v == max(list(data.values()))] 

if len(Discharge_mode) == n: 
    Discharge_mode = "NA"
elif len(Discharge_mode) > 1:
    Discharge_mode = statistics.median_low(Discharge_mode)
else: 
    Discharge_mode = Discharge_mode[0]
     
Discharge_stats = {'Qualifier': ['Min', 'Max', 'Median', 'Mode', 'Percent_25' ,'Percent_75'], 
                   'Discharge_cfps': [Discharge_min, Discharge_max, Discharge_median, Discharge_mode, Discharge_25, Discharge_75]}
Discharge_stats = pd.DataFrame(Discharge_stats)

Discharge_target = Discharge.merge(Discharge_stats, left_on='Discharge_cfps', right_on='Discharge_cfps')
Discharge_target = Discharge_target[['year', "month", 'day', 'Discharge_cfps', "Qualifier"]].drop_duplicates()
Discharge_target.to_csv(In + "\\Target_imagery\\" + Loc + "\\Discharge_Stats_Representative_Imagery_" + Loc + ".csv", index=False)

#%% Plot Discharge Histogram with selected stats
Discharge_stats_t = Discharge_stats.transpose()
Discharge_stats_t.columns = Discharge_stats_t.iloc[0]
Discharge_stats_t = Discharge_stats_t[1:]

plt.hist(Discharge['Discharge_cfps'], bins=50, color='skyblue', edgecolor='black')
plt.axvline(Discharge_stats_t['Max'].iloc[0], color='orange', linestyle='dashed', linewidth=1, label ='Max')
plt.axvline(Discharge_stats_t['Min'].iloc[0], color='red', linestyle='dashed', linewidth=1, label ='Min')
plt.axvline(Discharge_stats_t['Median'].iloc[0], color='purple', linestyle='dashed', linewidth=1, label ='Median')
plt.axvline(Discharge_stats_t['Mode'].iloc[0], color='magenta', linestyle='dashed', linewidth=1, label ='Mode')
plt.axvline(Discharge_stats_t['Percent_25'].iloc[0], color='blue', linestyle='dashed', linewidth=1, label ='Percent_25')
plt.axvline(Discharge_stats_t['Percent_75'].iloc[0], color='green', linestyle='dashed', linewidth=1, label ='Percent_75')
 
# Adding labels and title
plt.xlabel('Discharge [cfps]')
plt.ylabel('Frequency')
plt.title('Discharge Distribution of ' + Loc + '; Descriptive Stats')
plt.legend(facecolor='grey', framealpha=1, loc='upper right')
 
# Display the plot
plt.savefig(Out + "\\Stats\\Discharge_Distribution_" + Loc + "_Stats.png")
plt.show()

#%%
def roundup(x):
    return int(math.ceil(x / 100.0)) * 100

def rounddown(x):
    return int(math.floor(x / 100.0)) * 100

eqint = np.linspace(roundup(Discharge_min), rounddown(Discharge_max), num = 11, dtype = int)
splits = np.linspace(0, 100, num = 11)
splits = [int(splits) for splits in splits]

nearest_int = []
for i in range(len(eqint)):
    test = min(Discharge['Discharge_cfps'], key=lambda x:abs(x-eqint[i]))
    nearest_int.append(test)
    
Discharge_eqint = pd.DataFrame(list(zip(splits, nearest_int)), columns=['Qualifier', 'Discharge_cfps'])

splits = [str(splits) for splits in splits]

Discharge_eqint_target = Discharge.merge(Discharge_eqint, left_on='Discharge_cfps', right_on='Discharge_cfps')
Discharge_eqint_target = Discharge_eqint_target[['year', "month", 'day', 'Discharge_cfps', "Qualifier"]].drop_duplicates()
Discharge_eqint_target.to_csv(In + "\\Target_imagery\\" + Loc + "\\Discharge_Eqint_Representative_Imagery_" + Loc + ".csv", index=False)

#%% Plot Discharge Histogram with equal int
Discharge_eqint_t = Discharge_eqint.transpose()
Discharge_eqint_t.columns = Discharge_eqint_t.iloc[0]
Discharge_eqint_t = Discharge_eqint_t[1:]

plt.hist(Discharge['Discharge_cfps'], bins=50, color='skyblue', edgecolor='black')
plt.axvline(Discharge_eqint_t.iloc[0,0], color='red', linestyle='dashed', linewidth=1, label ='Min')
plt.axvline(Discharge_eqint_t.iloc[0,1], color='green', linestyle='dashed', linewidth=1)
plt.axvline(Discharge_eqint_t.iloc[0,2], color='green', linestyle='dashed', linewidth=1)
plt.axvline(Discharge_eqint_t.iloc[0,3], color='green', linestyle='dashed', linewidth=1)
plt.axvline(Discharge_eqint_t.iloc[0,4], color='green', linestyle='dashed', linewidth=1)
plt.axvline(Discharge_eqint_t.iloc[0,5], color='green', linestyle='dashed', linewidth=1)
plt.axvline(Discharge_eqint_t.iloc[0,6], color='green', linestyle='dashed', linewidth=1)
plt.axvline(Discharge_eqint_t.iloc[0,7], color='green', linestyle='dashed', linewidth=1)
plt.axvline(Discharge_eqint_t.iloc[0,8], color='green', linestyle='dashed', linewidth=1)
plt.axvline(Discharge_eqint_t.iloc[0,9], color='green', linestyle='dashed', linewidth=1)
plt.axvline(Discharge_eqint_t.iloc[0,10], color='orange', linestyle='dashed', linewidth=1, label ='Max')

# Adding labels and title
plt.xlabel('Discharge [cfps]')
plt.ylabel('Frequency')
plt.title('Discharge Distribution of ' + Loc + '; Equal Intervals')
plt.legend(facecolor='grey', framealpha=1, loc='upper right')
 
# Display the plot
plt.savefig(Out + "\\Eqint\\Discharge_Distribution_" + Loc + "_EqInt.png")
plt.show()

