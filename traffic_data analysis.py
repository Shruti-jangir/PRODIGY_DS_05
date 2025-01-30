import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from datetime import datetime
import json

# Load Datasets
data_cleaned = pd.read_csv(r"C:\Users\shrut\Downloads\archive (12)\cleaned.csv")
data_raw = pd.read_csv(r"C:\Users\shrut\Downloads\archive (12)\RTA Dataset.csv")

print("Loaded 'cleaned.csv' and 'RTA Dataset.csv' successfully.")
data = data_cleaned
print("Using the cleaned dataset.")

print("Dataset Preview:")
print(data.head())

# Data Preparation
data_raw["Time"] =  pd.to_datetime(data_raw["Time"], errors="coerce")
data_raw["Hour"] = data_raw["Time"].dt.hour

data_raw["Accident_severity"] = data_raw["Accident_severity"].astype(str)
data_cleaned["Accident_severity"] = data_cleaned["Accident_severity"].astype(str)

data_raw.fillna("Unknown", inplace=True)
data_cleaned.fillna("Unknown", inplace=True)

print("Columns after preparation:")
print(data.columns)

# Descriptive Satistics
weather_counts = data['Weather_conditions'].value_counts()
road_type_counts = data['Road_surface_type'].value_counts()

data_raw["Time"] = pd.to_datetime(data_raw["Time"], errors="coerce")
data_raw["hour"] = data_raw["Time"].dt.hour

print(data_raw["hour"].head())

time_counts = data_raw['hour'].value_counts().sort_index()

day_counts = data_raw['Day_of_week'].value_counts()

light_counts= data_raw['Light_conditions'].value_counts()

print("\nAccident Counts by weather:")
print(weather_counts)
print("\nAccident Counts by Road Conditions:")
print(road_type_counts)
print("\nAccident Counts by Hour:")
print(time_counts)
print("\nAccident Counts by Day of the Week:")
print(day_counts)
print("\nAccident Counts by Month:")
print(light_counts)

# Select relevant columns for merging
common_columns = [
    "Age_band_of_driver", "Sex_of_driver", "Educational_level", "Driving_experience", 
    "Lanes_or_Medians", "Types_of_Junction", "Road_surface_type", "Light_conditions", 
    "Weather_conditions", "Type_of_collision", "Vehicle_movement", "Pedestrian_movement", 
    "Cause_of_accident", "Accident_severity"
]

df_merged = pd.concat([data_cleaned[common_columns], data_raw[common_columns + ["Hour", "Day_of_week"]]], ignore_index=True)

# visualizations
sns.set_style("whitegrid")

# Bar Chart 1
plt.figure(figsize=(10, 5))
sns.barplot(x=weather_counts.index, y=weather_counts.values)
plt.title("Accidents by Weather Conditions")
plt.xlabel("Weather")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Bar Chart 2
plt.figure(figsize=(10, 5))
sns.barplot(x=road_type_counts.index, y=road_type_counts.values)
plt.title("Accidents by Road Conditions")
plt.xlabel("Road Condition")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Line Plot
plt.figure(figsize=(10, 5))
plt.plot(time_counts.index, time_counts.values, marker='o')
plt.title("Accidents by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Accidents")
plt.grid()
plt.tight_layout()
plt.show()

# Bar chart 3
plt.figure(figsize=(10, 5))
sns.barplot(x=day_counts.index, y=day_counts.values, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title("Accidents by Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.show()

#Bar chart 4
plt.figure(figsize=(10, 5))
sns.barplot(x=light_counts.index, y=light_counts.values)
plt.title("Accidents by Month")
plt.xlabel("Month")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Correlation Analysis
severity_weather = data.groupby(['Weather_conditions', 'Accident_severity']).size().unstack().fillna(0)
severity_road = data.groupby(['Road_surface_type', 'Accident_severity']).size().unstack().fillna(0)

plt.figure(figsize=(10, 5))
sns.heatmap(severity_weather, annot=True, fmt=".0f", cmap="coolwarm")
plt.title("Severity vs Weather")
plt.xlabel("Severity")
plt.ylabel("Weather")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(severity_road, annot=True, fmt=".0f", cmap="coolwarm")
plt.title("Severity vs Road Conditions")
plt.xlabel("Severity")
plt.ylabel("Road Condition")
plt.tight_layout()
plt.show()

# Save Summary Statistics
summary_stats = {
    "Weather Counts": weather_counts.to_dict(),
    "Road Condition Counts": road_type_counts.to_dict(),
    "Hourly Accident Counts": time_counts.to_dict(),
    "Daily Accident Counts": day_counts.to_dict(),
    "Monthly Accident Counts": light_counts.to_dict()    
}

with open("summart_stats.json", "w") as f:
    json.dump(summary_stats, f, indent=4)
    
print("Analysis and visualization completed. Summary Statistics saved as 'summary_stats.json.")