#load modules
from IPython import get_ipython
get_ipython().magic('reset -sf')

# C:\Users\OCRONING\AppData\Local\Programs\Spyder\Python\python -m pip install rasterio
import os
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from rasterio.plot import reshape_as_image

#%% Parameters
Loc = "LD1"
Type = "Eqint"
fp = r"C:\\Users\\OCRONING\\OneDrive - Environmental Protection Agency (EPA)\\Profile\\Documents\\PFAS_Foam\\"
In = fp + r"Inputs\\"
Out = fp + "Outputs\\" + Loc + "\\" + Type + "\\Unsupervised\\"

River_Clip_path = In + r"SkySat_Imagery\\" + Loc + "\\" + Loc + "_River_Clip\\" + Type + "\\"
Raster_name = os.listdir(River_Clip_path)

#%% Open clipped rasters
Dates = []    
for i in range(len(Raster_name)):
    Dates.append(Raster_name[i][0:8:1])
    
Time = []    
for i in range(len(Raster_name)):
    Time.append(Raster_name[i][9:15:1])
    
Raster_img = []
for i in range(len(Raster_name)):
    x = rio.open(River_Clip_path + Raster_name[i])
    Raster_img.append(x)
    
with rio.open(River_Clip_path + Raster_name[i]) as src:
    green_band = src.read(2)
    red_band = src.read(3)
    nir_band = src.read(4)
    kwargs = src.meta
    kwargs.update(dtype=rio.float32, count = 1)
    
#%% Obtain and Filter Dates
Month = []    
for i in range(len(Raster_name)):
    Month.append(int(Raster_name[i][4:6:1]))
    
Day = []    
for i in range(len(Raster_name)):
    Day.append(int(Raster_name[i][6:8:1]))

#%% Read in Discharge Metadata
Discharge = pd.read_csv(In + "\\Target_Imagery\\" + Loc + "\\Planet_Target_Imagery_" + Type + "_" + Loc + ".csv")

Discharge_meta = []
for i in range(len(Raster_name)):
    rslt_df = Discharge.loc[(Discharge['month'] == Month[i])
                            & (Discharge['day'] == Day[i])]
    Discharge_meta.append(rslt_df)

Discharge_dates = pd.concat(Discharge_meta)
Discharge_cfps = Discharge_dates['Actual_discharge'].tolist()

#%% Create unsupervised classification with kmeans

with rio.open(River_Clip_path + Raster_name[4]) as src:
    img = src.read()
    
# Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
reshaped_img = reshape_as_image(img)
bands, rows, cols = img.shape
    
k = 6 # num of clusters
    
kmeans_predictions = KMeans(n_clusters=k, random_state=0).fit(reshaped_img.reshape(-1, 4))
kmeans_predictions_2d = kmeans_predictions.labels_.reshape(rows, cols)

#%% Apply classification schema to rest of imagery and plot results
counts_list = []

for i in range(len(Raster_img)):
    
    with rio.open(River_Clip_path + Raster_name[i]) as src: img = src.read()
        
    # Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
    reshaped_img = reshape_as_image(img)
    bands, rows, cols = img.shape
                
    kmeans_predict = kmeans_predictions.predict(reshaped_img.reshape(-1, 4))
    kmeans_predictions_2d = kmeans_predict.reshape(rows, cols)
    
    #Plot classified raster
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize=(10,7))
    plt.imshow(kmeans_predictions_2d)
    plt.title('Discharge: ' + str(Discharge_cfps[i]) + " cfps; k = " + str(k) , fontsize=15, y = 1)
    plt.savefig(Out + "\\" + Loc + "_d" + str(Discharge_cfps[i]) + "_" + str(k) + "kmeans_classify.png")
    plt.show()
    
    im = plt.imshow(kmeans_predictions_2d)
    colours = im.cmap(im.norm(np.unique(kmeans_predictions_2d)))
    plt.close()
                  
    #Export classified raster
    with rio.open(In + 'SkySat_Imagery\\' + Loc + '\\' + Loc + '_Unsupervised\\' + Loc + "_d" + str(Discharge_cfps[i]) + '_kmeans.tif', 'w', **kwargs) as dst:
            dst.write_band(1, kmeans_predictions_2d.astype(rio.float32))
    
    #Get pixel counts of each land cover type
    counts = np. unique(kmeans_predictions_2d, return_counts=True)
    counts_df = pd.DataFrame(counts).transpose()
    counts_df.columns = ['LC_code', 'ncells']
    counts_df['area_m2'] = counts_df['ncells'] * (3*3)
    counts_df['discharge'] = Discharge_cfps[i]
    counts_list.append(counts_df)

Counts_df = pd.concat(counts_list)

#%% Summarize pixel land cover type from unsupervised alg (change if n clusters h=change, works for 6 classes)
conditions = [
    (Counts_df['LC_code'] == 0), #nan
    (Counts_df['LC_code'] == 1), #shallow water
    (Counts_df['LC_code'] == 2), #forest 
    (Counts_df['LC_code'] == 3), #foam
    (Counts_df['LC_code'] == 4), #water
    (Counts_df['LC_code'] == 5), #forest edge
]

values = ['a', 'b', 'c', 'd', 'e', 'f']
Counts_df['LC_class'] = np.select(conditions, values)

Counts_df.to_csv(Out + "\\Unsupervised_LC_counts.csv", index=False)

#%% Plot discharge vs foam area
Class = list(set(Counts_df['LC_code'].values.tolist()))
Counts_df = Counts_df.reset_index()
Counts_df = Counts_df.drop(['index'], axis=1)

for i in range(len(Class)):
    Class_df = Counts_df[Counts_df['LC_code'] == Class[i]]
    plt.plot(Class_df['discharge'], Class_df['area_m2'], marker='o', color = colours[i])

plt.ylim(0, 40000)
plt.xlabel('Discharge [cfps]')
plt.ylabel('Foam area [m2]')
plt.title('Area of Land Cover Type at ' + Loc + ' Dam at Varying Discharge Levels\nUnsupervised Classifier')
plt.savefig(Out + "\\" + Loc + "_Land_Cover_Type_Area.png")

plt.show()