#clear environment
rm(list=ls())

#Load library
library(dataRetrieval)

#Set filepaths
Out <- "C:/Users/OCRONING/OneDrive - Environmental Protection Agency (EPA)/Profile/Documents/PFAS_Foam/Foam_Classification/Inputs/Discharge/"
Loc <- "LD1" #WOH or LD1

#Date parameter
Start_date <- "2023-01-01"
End_date <- Sys.Date() #Today's date

#Damn gage number
if (Loc == "WOH"){
  siteNumber <- "02105500"
} else if (Loc == "LD1"){
  siteNumber <- "02105769"
} else {
  print("Provide USGS Gage Number")
}

#Damn gage meta  
Info <- readNWISsite(siteNumber)

# Raw daily data
parameterCd <- "00060" #Daily discharge code

rawDailyData <- readNWISdv(
  siteNumber, parameterCd,
  Start_date, End_date
)

#Filter and rename columns
rawDailyData <- rawDailyData[,c(1:4)]
names(rawDailyData) <- c("agency_cd", "site_no", "Date", "Discharge_cfps")

#Split Date into yyyy mm dd columns
rawDailyData$year <- substr(rawDailyData$Date,1,4)
rawDailyData$month <- substr(rawDailyData$Date,6,7)
rawDailyData$day <- substr(rawDailyData$Date,9,10)

#Export
write.csv(rawDailyData, paste0(Out, "Discharge_", Loc, ".csv"), row.names = F)
write.csv(Info, paste0(Out, "Discharge_", Loc, "_meta.csv"), row.names = F)
