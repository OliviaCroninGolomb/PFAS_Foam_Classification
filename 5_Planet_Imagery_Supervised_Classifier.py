#load modules
from IPython import get_ipython
get_ipython().magic('reset -sf')

# C:\Users\OCRONING\AppData\Local\Programs\Spyder\Python\python -m pip install rasterio
import os
import rasterio as rio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import mapping
import pandas as pd
from rasterio.plot import reshape_as_image

#%% Parameters
Loc = "LD1"
Type = "Stats"
fp = r"C:\\Users\\OCRONING\\OneDrive - Environmental Protection Agency (EPA)\\Profile\\Documents\\PFAS_Foam\\"
In = fp + r"Inputs\\"
Out = fp + "Outputs\\" + Loc + "\\" + Type + "\\Supervised\\"

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

#%% Import LC Class Shapefiles
water = gpd.read_file(In + r'Shapefiles\\Land_Cover_Training\\' + Loc + '\\Water.shp')
foam = gpd.read_file(In + r'Shapefiles\\Land_Cover_Training\\' + Loc + '\\Foam.shp')
forest = gpd.read_file(In + r'Shapefiles\\Land_Cover_Training\\' + Loc + '\\Forest_Water.shp')
urban = gpd.read_file(In + r'Shapefiles\\Land_Cover_Training\\' + Loc + '\\Urban_Water.shp')

water['classname'] = 'Water'
water['classvalue'] = 1
foam['classname'] = 'Foam'
foam['classvalue'] = 2
forest['classname'] = 'Forest'
forest['classvalue'] = 3
urban['classname'] = 'Urban'
urban['classvalue'] = 4

#%% Combine LC Class shapefiles into one shapefile
LC_training = gpd.GeoDataFrame(pd.concat([water, foam, forest, urban]))
print(LC_training.crs)
print(len(LC_training))
LC_training_filtered = LC_training.filter(['classname', 'classvalue', 'geometry'])
LC_training_filtered.head()
Index = list(range(0,len(LC_training_filtered),1))
LC_training_filtered["test"] = Index
LC_training_filtered = LC_training_filtered.set_index('test')
LC_training_filtered.index.names = ['index']

LC_training_filtered.to_file(In + r'Shapefiles\\Land_Cover_Training\\' + Loc + '\\' + Loc + '_LC_Training.shp')

#%% Extract the raster values within the polygon 
geoms = LC_training_filtered.geometry.values 

X = np.array([], dtype=np.int8).reshape(0,4) # pixels for training
y = np.array([], dtype=np.string_) # labels for training

with (Raster_img[1]) as src:
    band_count = src.count
    for index, geom in enumerate(geoms):
        feature = [mapping(geom)]
        # the mask function returns an array of the raster pixels within this feature
        out_image, out_transform = mask(src, feature, crop=True) 
        # eliminate all the pixels with 0 values for all 8 bands - AKA not actually part of the shapefile
        out_image_trimmed = out_image[:,~np.all(out_image == 0, axis=0)]
        # reshape the array to [pixel count, bands]
        out_image_reshaped = out_image_trimmed.reshape(-1, band_count)
        # append the labels to the y array
        y = np.append(y,[LC_training_filtered["classname"][index]] * out_image_reshaped.shape[0]) 
        # stack the pizels onto the pixel array
        X = np.vstack((X,out_image_reshaped))  

# What are our classification labels?
labels = np.unique(LC_training_filtered["classname"])
print('The training data include {n} classes: {classes}\n'.format(n=labels.size, classes=labels))

# We will need a "X" matrix containing our features, and a "y" array containing our labels
print('Our X matrix is sized: {sz}'.format(sz=X.shape))
print('Our y array is sized: {sz}'.format(sz=y.shape))

#%% Plot intensities of each band 
fig, ax = plt.subplots(figsize=[5,8])

# numbers 1-5
band_count = np.arange(1,5)

classes = np.unique(y)
for class_type in classes:
    band_intensity = np.mean(X[y==class_type, :], axis=0)
    ax.plot(band_count, band_intensity, label=class_type)

# plot them as lines
ax.set_xlabel('Band #')
ax.set_ylabel('Reflectance Value')
ax.set_xticks(range(1, 5))
ax.set
ax.legend(loc="upper left")
ax.set_title('Band Intensities Full Overview')

plt.savefig(Out + "\\" + Loc + "_Supervised_Band_Intensities.png", bbox_inches='tight')

#%% Reclassify string class to integers
def str_class_to_int(class_array):
    class_array[class_array == 'Foam'] = 0
    class_array[class_array == 'Water'] = 1
    class_array[class_array == 'Forest'] = 2
    class_array[class_array == 'Urban'] = 3
    return(class_array.astype(int))

#%% Prepare training and testing data
from sklearn.metrics import accuracy_score#, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%% Gaussian Native Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_y_pred = gnb.predict(X_test) 

gnb_accuracy = accuracy_score(y_test, gnb_y_pred)
print("GNB Accuracy:", round(gnb_accuracy,2))

#%% Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
knn_y_pred = knn.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_y_pred)
print("KNN Accuracy:", round(knn_accuracy,2))

#%% Support Vector Machine
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)
svc_y_pred = svc.predict(X_test)

svc_accuracy = accuracy_score(y_test, svc_y_pred)
print("SVC Accuracy:", round(svc_accuracy,2))

#%% Random Forest
from sklearn.ensemble import RandomForestClassifier

param_dist = {'n_estimators': randint(50,1000),
              'max_depth': randint(1,30)}

rf = RandomForestClassifier()
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

rand_search.fit(X_train, y_train)
best_rf = rand_search.best_estimator_

rf_y_pred = best_rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_y_pred)
print("RF Accuracy:", round(rf_accuracy,2))

#%% Classify imageery and plot results
predictions = [gnb, knn, svc, best_rf]
counts_all = []

for j in range(len(Raster_img)):

    with rio.open(River_Clip_path + Raster_name[j][:-4] + ".tif") as src:
        img = src.read()
    reshaped_img = reshape_as_image(img)
    
    prediction = []
    counts_list = []
    counts_all.append(counts_list)

    for i in range(len(predictions)):
        class_prediction = predictions[i].predict(reshaped_img.reshape(-1, 4))
        reshape_prediction = class_prediction.reshape(reshaped_img[:, :, 0].shape)
        int_prediction = str_class_to_int(reshape_prediction)
        prediction.append(int_prediction)

    def color_stretch(image, index):
        colors = image[:, :, index].astype(np.float64)
        for b in range(colors.shape[2]):
            colors[:, :, b] = rio.plot.adjust_band(colors[:, :, b])
        return colors
        
    # find the highest pixel value in the prediction image
    n = 3
    
    # next setup a colormap for our map
    colors = dict((
        (0, (244, 164, 96, 255)),   # Tan - foam
        (1, (48, 156, 214, 255)),   # Blue - Water
        (2, (96, 19, 134, 255)),    # Purple - Emergent Wetland
        (3, (139,69,19, 255)),      # Brown - WetSand
    ))
    
    # Put 0 - 255 as float 0 - 1
    for k in colors:
        v = colors[k]
        _v = [_v / 255.0 for _v in v]
        colors[k] = _v
        
    index_colors = [colors[key] if key in colors else 
                    (255, 255, 255, 0) for key in range(0, n+1)]
    
    cmap = plt.matplotlib.colors.ListedColormap(index_colors, 'Classification', n+1)

    accuracy_scores = [gnb_accuracy, knn_accuracy, svc_accuracy, rf_accuracy]
    
    fig, axs = plt.subplots(2,2,figsize=(10,7))
    axs[0,0].imshow(prediction[0], cmap=cmap, interpolation='none')
    axs[0,0].set_title("GNB, Accuracy:" + str(round(accuracy_scores[0],2)))
    
    axs[0,1].imshow(prediction[1], cmap=cmap, interpolation='none')
    axs[0,1].set_title("KNN, Accuracy:" + str(round(accuracy_scores[1],2)))
    
    axs[1,0].imshow(prediction[2], cmap=cmap, interpolation='none')
    axs[1,0].set_title("SVC, Accuracy:" + str(round(accuracy_scores[2],2)))
    
    axs[1,1].imshow(prediction[3], cmap=cmap, interpolation='none')
    axs[1,1].set_title("RT, Accuracy:" + str(round(accuracy_scores[3],2)))
    
    plt.subplots_adjust(wspace=-0.35, hspace=0.35)
    fig.suptitle('Discharge: ' + str(Discharge_cfps[j]) + " cfps", fontsize=15)
    plt.savefig(Out + "\\" + Loc + "_d" + str(Discharge_cfps[j]) + "_Supervised.png")
    plt.show()
    
    im = plt.imshow(prediction[1])
    colours = im.cmap(im.norm(np.unique(prediction[1])))
    plt.close()

#%%
prediction_name = ['gnb', 'knn', 'svc', 'best_rf']

counts_list = []
for i in range(len(prediction)):
    counts = np.unique(prediction[i], return_counts=True)
    counts_df = pd.DataFrame(counts).transpose()
    counts_df.columns = ['LC_code', 'ncells']
    counts_df['area_m2'] = counts_df['ncells'] * (3*3)
    counts_df['discharge'] = Discharge_cfps[1]
    counts_df['method'] = prediction_name[i]
    counts_list.append(counts_df)
#%% generate counts of pixels in each LC class
prediction_name = ['gnb', 'knn', 'svc', 'best_rf']

count_df1 = []
for i in range(len(counts_all)):
    
    count_df = counts_all[i]
    count_df = count_df[0]
    count_df1.append(count_df)

Counts_df = pd.concat(count_df1)

#%% Summarize pixel land cover type from unsupervised alg (change if n clusters h=change, works for 6 classes)
conditions = [
    (Counts_df['LC_code'] == 0), #nan
    (Counts_df['LC_code'] == 1), #shallow water
    (Counts_df['LC_code'] == 2), #forest 
    (Counts_df['LC_code'] == 3) #foam
]

values = ['Foam', 'Water', 'Forest', 'Urban']
Counts_df['LC_class'] = np.select(conditions, values)

Counts_df.to_csv(Out + "\\Supervised_LC_counts.csv", index=False)

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
plt.title('Area of Land Cover Type at ' + Loc + ' Dam at Varying Discharge Levels\nSupervised Classifier')
plt.savefig(Out + "\\" + Loc + "_Land_Cover_Type_Area.png")

plt.show()