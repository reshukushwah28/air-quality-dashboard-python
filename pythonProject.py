import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ——— CONFIG ———
#This stores the path to your CSV file. The r before the string makes it a raw string to handle backslashes properly in file paths.

CSV_PATH = r"C:\Users\kushw\OneDrive\Documents\Downloads\Air_Quality (1).csv"

# ——— 1) LOAD & CLEAN ———
# Loads the dataset into a DataFrame called df.
df = pd.read_csv(CSV_PATH)
#Cleans the column names by removing spaces and replacing them with underscores for easier access.
df.columns = df.columns.str.strip().str.replace(" ", "_")

#Drops any column that contains only NaN values.
df.dropna(axis=1, how="all", inplace=True)

#Fills any remaining NaN values with the previous value in the column (forward fill).
df.ffill(inplace=True)           # future‑proof over df.fillna(method="ffill")

df.drop_duplicates(inplace=True)

# ——— 2) 2) DESCRIPTIVE STATISTICS ———
#Selects all numeric columns and calculates statistics like mean, min, max, standard deviation, etc.
numeric_cols = df.select_dtypes(include=[np.number]).columns
desc_stats = df[numeric_cols].describe()
print("\n▶ Descriptive statistics for numeric columns:\n")
print(desc_stats, "\n")


# ——— 3) PIVOT TABLE (top 10) ———
#Creates a pivot table showing average pollution values (Data_Value) for each Measure (like PM2.5, CO) across different places.
pivot_data = df.pivot_table(
    values="Data_Value",
    index="Geo_Place_Name",
    columns="Measure",
    aggfunc="mean"
)
print("▶ Pivot table (average Data_Value) — top 10 places:\n")
print(pivot_data.head(10), "\n") #Prints the top 10 rows of the pivot table

# ——— 4) LINE PLOT ———
#Transposes the top 10 rows and creates a line plot showing trends of different measures across top 10 cities.
plt.figure(figsize=(10, 5))
sns.lineplot(data=pivot_data.head(10).T)
plt.title("Measures across Top 10 Locations")
plt.xlabel("Measure")
plt.ylabel("Average Data Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show() #Displays a clear line chart comparing pollution measures across cities.

# ——— 5) TOP 6 CITIES (pie chart) ———
#Displays a pie chart showing how data is distributed across the top 6 cities.
top_cities = df["Geo_Place_Name"].value_counts().head(6)
print("▶ Top 6 cities by number of records:\n")
print(top_cities, "\n")

plt.figure(figsize=(6, 6))
plt.pie(top_cities.values,
        labels=top_cities.index,
        autopct="%1.1f%%",
        startangle=140)
plt.title("Distribution of Top 6 Cities")
plt.axis("equal")
plt.tight_layout()
plt.show()

# ——— 6) TOP 10 POLLUTANTS (bar chart) ———
#Calculates and prints the average value of each pollutant, then shows the top 10.
top_pollutants = (df.groupby("Measure")["Data_Value"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(10))
print("▶ Top 10 pollutants by average value:\n")
print(top_pollutants, "\n")

plt.figure(figsize=(8, 6))
sns.barplot(x=top_pollutants.values, y=top_pollutants.index)
plt.title("Top 10 Pollutants by Average Value")
plt.xlabel("Average Data Value")
plt.ylabel("Pollutant")
plt.tight_layout()
plt.show() #Displays a horizontal bar chart of pollutants with highest average values.

# ——— 7) CORRELATION MATRIX ———
# Calculates and prints how much numeric columns are correlated with each other.
corr = df[numeric_cols].corr()
print("▶ Correlation matrix of numeric features:\n")
print(corr, "\n")

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show() 
#Shows a heatmap where:
#Red = Strong positive correlation
#Blue = Strong negative correlation

# ——— 8) SCENARIO COLUMNS & TOP CITY ———
# Creates two new columns:
#One assuming a 20% increase in pollution
#Another assuming a 20% decrease
df["Scenario_High"] = df["Data_Value"] * 1.2
df["Scenario_Low"]  = df["Data_Value"] * 0.8

city_means       = df.groupby("Geo_Place_Name")["Data_Value"].mean()
top_city         = city_means.idxmax()
top_city_avg_val = city_means.max()
# Calculates average pollution for each city and identifies the city with the highest average.

print(f"▶ City with highest average pollution: {top_city}")
print(f"  → Average pollution value: {top_city_avg_val:.2f}\n")
#Displays a few records for that city.

# Optional: show a sample of that city’s records
print(f"▶ Sample records for {top_city}:\n")
print(df[df["Geo_Place_Name"] == top_city].head(), "\n")

# ——— 9) TIME‑TREND (if you have a Date column) ———
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
    time_trend = df.groupby("Date")["Data_Value"].mean() 
    #If a Date column exists, it converts it to datetime and calculates average pollution per date.
    print("▶ First 10 average‑over‑time values:\n")
    print(time_trend.head(10), "\n") # Prints the first 10 time-based averages.
    
    plt.figure(figsize=(10, 4))
    plt.plot(time_trend.index, time_trend.values)
    plt.title("Average Pollution Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Average Data Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()  #Displays a line plot of pollution values over time.

# ——— 10) BOXPLOT BY MEASURE ———
#📊 Shows how pollution values vary for each measure:
#Box shows the spread (25%–75%)
#Line inside is the median
#Dots outside are outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="Measure", y="Data_Value")
plt.title("Boxplot of Data Value by Measure")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("✅ All outputs printed to terminal and plots displayed interactively.")
#Confirms that all processing and visualization steps were successful.
