from IPython import get_ipython
get_ipython().magic('reset -sf')

#load modules
# C:\Users\OCRONING\AppData\Local\Programs\Spyder\Python\python -m pip install rasterio
import os
import rasterio as rio
from rasterio.mask import mask
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

#%%Parameters
Loc = "LD1"
Type = "Stats"
fp = r"C:\\Users\\OCRONING\\OneDrive - Environmental Protection Agency (EPA)\\Profile\\Documents\\PFAS_Foam\\"
In = fp + r"Inputs\\"
Out = fp + "Outputs\\" + Loc + "\\" + Type + "\\"

Raster_path = In + r"SkySat_Imagery\\" + Loc + "\\" + Loc + "_Imagery\\" + Type + "\\"
River_Clip_path = In + r"SkySat_Imagery\\" + Loc + "\\" + Loc + "_River_Clip\\" + Type + "\\"
Raster_name = os.listdir(Raster_path)

#%% Open raw rasters
Dates = []    
for i in range(len(Raster_name)):
    Dates.append(Raster_name[i][0:8:1])
    
Time = []    
for i in range(len(Raster_name)):
    Time.append(Raster_name[i][9:15:1])
    
Raster_img = []
for i in range(len(Raster_name)):
    x = rio.open(Raster_path + Raster_name[i])
    Raster_img.append(x)

#%% Clip Rasters
AOI_rivers = gpd.read_file(In + "Shapefiles\\" + Loc + "\\" + Loc + "_Resolvable_Rivers.shp")

fig, ax = plt.subplots(figsize=(6, 6))
AOI_rivers.plot(ax=ax)
ax.set_title("Resolvable Rivers Crop Extent", fontsize=16)
plt.show()

#%%
for i in range(len(Raster_name)):

    outRas = River_Clip_path + Raster_name[i][:-4] + "_" + Loc + "_River.tif"
    
    with rio.open(Raster_path + Raster_name[i]) as src:
        AOI_rivers=AOI_rivers.to_crs(src.crs)
        out_image, out_transform=mask(src,AOI_rivers.geometry,crop=True)
        out_meta=src.meta.copy() # copy the metadata of the source DEM
    out_meta.update({
        "driver":"Gtiff",
        "height":out_image.shape[1], # height starts with shape[1]
        "width":out_image.shape[2], # width starts with shape[2]
        "transform":out_transform
    })
    with rio.open(outRas,'w',**out_meta) as dst:
        dst.write(out_image)

#%% Read Clipped Rasters
Raster_name = os.listdir(River_Clip_path)

Raster_img = []
for i in range(len(Raster_name)):
    x = rio.open(River_Clip_path + Raster_name[i])
    Raster_img.append(x)
    
#%% Obtain and Filter Discharge
Year = []    
for i in range(len(Raster_name)):
    Year.append(int(Raster_name[i][0:4:1]))

Month = []    
for i in range(len(Raster_name)):
    Month.append(int(Raster_name[i][4:6:1]))
    
Day = []    
for i in range(len(Raster_name)):
    Day.append(int(Raster_name[i][6:8:1]))
    
Hour = []    
for i in range(len(Raster_name)):
    Hour.append(int(Raster_name[i][9:11:1]))

def myround(x, base=15):
    return base * round(x/base)

Minute = []    
for i in range(len(Raster_name)):
    minute = (Raster_name[i][11:13:1])
    min_round = myround(int(minute), base = 15)
    Minute.append(min_round)
    
#%%
Discharge = pd.read_csv(In + "\\Target_Imagery\\" + Loc + "\\Planet_Target_Imagery_" + Type + "_" + Loc + ".csv")

Discharge_meta = []
for i in range(len(Raster_name)):
    rslt_df = Discharge.loc[(Discharge['year'] == Year[i])
                            & (Discharge['month'] == Month[i])
                            & (Discharge['day'] == Day[i])]
    Discharge_meta.append(rslt_df)

Discharge_dates = pd.concat(Discharge_meta)
Discharge_cfps = Discharge_dates['Actual_discharge'].tolist()

#%% Plot Bands
for i in range(len(Raster_img)):

    fig, axs = plt.subplots(2,2,figsize=(15,7))
    red_band = axs[0,0].imshow(Raster_img[i].read(3), cmap="Reds")
    axs[0,0].set_title("Red Band")
    green_band = axs[0,1].imshow(Raster_img[i].read(2), cmap="Greens")
    axs[0,1].set_title("Green Band")
    blue_band = axs[1,0].imshow(Raster_img[i].read(1), cmap="Blues")
    axs[1,0].set_title("Blue Band")
    NIR_band = axs[1,1].imshow(Raster_img[i].read(4), cmap="Purples")
    axs[1,1].set_title("NIR Band")
    
    fig.suptitle('Discharge: ' + str(Discharge_cfps[i]) + " cfps", fontsize=15)
    
    plt.subplots_adjust(wspace=-0.7, hspace=0.3)
    plt.savefig(Out + "\\" + Loc + "_d" + str(Discharge_cfps[i]) + "_Bands.png")
    plt.show()

del(green_band, blue_band, red_band, NIR_band)

#%% NDVI NDWI
for i in range(len(Raster_img)):

    with rio.open(River_Clip_path + Raster_name[i]) as src:
        green_band = src.read(2)
        red_band = src.read(3)
        nir_band = src.read(4)
        
    ndwi = (green_band.astype(float) - nir_band.astype(float)) / (green_band.astype(float) + nir_band.astype(float))
    ndvi = (nir_band.astype(float) - red_band.astype(float)) / (red_band.astype(float) + nir_band.astype(float))
    
    #Plot NDVI and NDWI
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(10,7))
    
    ndwi_plot = axs[0].imshow(ndwi, cmap="RdYlGn", vmin=-0.8, vmax=0.4)
    axs[0].set_title("NDWI")
    plt.colorbar(ndwi_plot, fraction=0.046, pad=0.04)
    
    ndvi_plot = axs[1].imshow(ndvi, cmap="RdYlGn", vmin=-0.1, vmax=0.9)
    axs[1].set_title("NDVI")
    plt.colorbar(ndvi_plot, fraction=0.046, pad=0.04)
    
    fig.suptitle('Discharge: ' + str(Discharge_cfps[i]) + " cfps", fontsize=15, y = 0.95)
    
    plt.subplots_adjust(wspace=0.5, hspace=0.3)
    plt.savefig(Out + "\\" + Loc + "_d" + str(Discharge_cfps[i]) + "_ndvi_ndwi.png")
    plt.show()
    
    # Export Rasters
    kwargs = src.meta
    kwargs.update(dtype=rio.float32, count = 1)
    
    with rio.open(In + "SkySat_Imagery\\" + Loc + "\\" + Loc + "_Indexes\\" + Type + "\\" + Loc + "_d" + str(Discharge_cfps[i]) + "_ndvi.tif", 'w', **kwargs) as dst:
            dst.write_band(1, ndvi.astype(rio.float32))
            
    with rio.open(In + "SkySat_Imagery\\" + Loc + "\\" + Loc + "_Indexes\\"+ Type + "\\" + Loc + "_d" + str(Discharge_cfps[i]) + "_ndwi.tif", 'w', **kwargs) as dst:
            dst.write_band(1, ndwi.astype(rio.float32))