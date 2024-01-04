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
import geopandas as gpd
import math

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
with rio.open(River_Clip_path + Raster_name[1]) as src:
    img = src.read()
    
# Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
reshaped_img = reshape_as_image(img)
bands, rows, cols = img.shape
    
k = 6 # num of clusters
    
kmeans_predictions = KMeans(n_clusters=k, random_state=0)
kmeans_predictions.fit(reshaped_img.reshape(-1, 4))
kmeans_predictions_2d = kmeans_predictions.labels_.reshape(rows, cols)

fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize=(10,7))
plt.imshow(kmeans_predictions_2d)
plt.title('Discharge: ' + str(Discharge_cfps[4]) + " cfps; k = " + str(k) , fontsize=15, y = 1)
plt.show()

im = plt.imshow(kmeans_predictions_2d)
colours = im.cmap(im.norm(np.unique(kmeans_predictions_2d)))
plt.close()

#%% Plot clusters
X = reshaped_img.reshape(-1, 4)
y_kmeans = kmeans_predictions.predict(reshaped_img.reshape(-1, 4))

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans_predictions.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.title('KMeans Clusters\nDischarge: ' + str(Discharge_cfps[4]) + " cfps; k = " + str(k) , fontsize=15, y = 1)
    
#%% Classify Rasters
for i in range(len(Raster_img)):
    
    with rio.open(River_Clip_path + Raster_name[i]) as src: 
        img = src.read()
        
    # Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
    reshaped_img = reshape_as_image(img)
    bands, rows, cols = img.shape
    
    kmeans_predictions = KMeans(n_clusters=k, random_state=0)
    kmeans_predictions.fit(reshaped_img.reshape(-1, 4))
    kmeans_predictions_2d = kmeans_predictions.labels_.reshape(rows, cols)
       
    #kmeans_predict = kmeans_predictions.predict(reshaped_img.reshape(-1, 4))
    #kmeans_predictions_2d = kmeans_predict.reshape(rows, cols)
    
    #Plot classified raster
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize=(10,7))
    plt.imshow(kmeans_predictions_2d)
    plt.title('Discharge: ' + str(Discharge_cfps[i]) + " cfps; k = " + str(k) , fontsize=15, y = 1)
    plt.savefig(Out + "\\" + Loc + "_d" + str(Discharge_cfps[i]) + "_" + str(k) + "kmeans_classify.png")
    plt.show()
    
    #Export classified raster
    with rio.open(In + 'SkySat_Imagery\\' + Loc + '\\' + Loc + '_Unsupervised\\' + Type + "\\" + Loc + "_d" + str(Discharge_cfps[i]) + '_kmeans.tif', 'w', **kwargs) as dst:
            dst.write_band(1, kmeans_predictions_2d.astype(rio.float32))

#%% Extract foam area
Class_fp = In + 'SkySat_Imagery\\' + Loc + '\\' + Loc + '_Unsupervised\\' + Type + "\\"
Ras_Class = [file for file in os.listdir(Class_fp)if file.endswith(".tif")]
pointData = gpd.read_file(In + "Shapefiles\\Land_Cover_Training\\" + Loc + "\\Foam_Point.shp")

foam_list = []

for i in range(len(Ras_Class)):  

    Class_img = rio.open(Class_fp + Ras_Class[i])
    
    for point in pointData['geometry']:
        x = point.xy[0][0]
        y = point.xy[1][0]
        row, col = Class_img.index(x,y)
        foam_value = Class_img.read(1)[row,col]
    
    with rio.open(Class_fp + Ras_Class[i]) as src: 
         img = src.read()
         
    reshaped_img = reshape_as_image(img)
    bands, rows, cols = img.shape
    
    counts = np.unique(reshaped_img, return_counts=True)
    counts_df = pd.DataFrame(counts).transpose()
    counts_df.columns = ['LC_code', 'ncells']
    counts_df['area_m2'] = counts_df['ncells'] * (3*3)
    counts_df['discharge'] = Discharge_cfps[i]
    
    foam_df = counts_df[counts_df['LC_code'] == foam_value] 
    foam_list.append(foam_df)

Foam_df = pd.concat(foam_list)
Foam_df.to_csv(Out + "\\Unsupervised_Foam_Counts.csv", index=False)

#%% Plot discharge vs foam area
max_ylim = max(Foam_df['area_m2'])
max_ylim = math.ceil(max_ylim/10000)*10000

plt.scatter(Foam_df.discharge, Foam_df.area_m2)

plt.ylim(0, max_ylim)
plt.xlabel('Discharge [cfps]')
plt.ylabel('Foam area [m2]')
plt.title('Foam Area at ' + Loc + ' Dam at Varying Discharge Levels\nSupervised Classifier')

plt.savefig(Out + "\\" + Loc + "_Foam_Area.png")
plt.show()