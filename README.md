# Project 4 | CT Opioid Crisis Exploration | Unsupervised Machine Learning
Project 4 Penn Data Bootcamp | Jillian Walter, Carleigh West, Marina Connelly, Kehleboe Gongloe

**Background:**

This project explores patterns in Drug Overdose deaths from 2018-2023 in the state of Connecticut using an Unsupervised Machine Learning Model. This data was sourced from the Office of the Chief Medical Examiner Statistics site (https://portal.ct.gov/ocme/statistics). The data contains detailed information around each individuals death due to drug overdose, with each row representing one individual (7.5k individuals reported). Data compiled included the below:

- Date of Death
- Gender (Male, Female, Other)
- Race
- Location of overdose death (city, county, state, zip code)
- Location of Injury (ie. Home, Others' Residence, Outdoors, Vehicle, etc.)
- _Binary values representing whether or not each of the below drugs were detected in the individuals postmortem toxicology test:_
- Heroin
- Cocaine
- Fetanyl
- Fetanyl Analogue
- Oxycodone
- Oxymorphone
- Ethanol
- Hydrocodone
- Benzodiazepines
- Methadone
- Meth/Amphetamine
- Tramadol
- Hydromorphone
- Morphine (not heroin)

The purpose of this exploration is to identify patterns in drug usage across different demographic clusters identified by our K-Means Machine Learning Model, to help prevent future deaths.

# Uploading into PostgreSQL and Jupyter Notebook:

Data in our initial download was in an excel format with different tabs per year; some years had excess columns and/or differing orders per columns, plus typos in some cells, so it was important to streamline the format into one cohesive CSV tab. The cleaned and condensed CSV was uploaded to PostgreSQL, with "ID" as the primary key for each individual row to represent each individual.

In Jupyter Notebook, we started by importing our necessary dependencies for the whole analysis:
- import pandas as pd
- import sqlite3
- import hvplot.pandas
- from pathlib import Path
- from sklearn.cluster import KMeans
- from sklearn.preprocessing import StandardScaler

our team utilized a SQLite file to read in the data, connecting to the database and reading in the data using the below commands:
- db_name = "opioid_analysis.sqlite"
- conn = sqlite3.connect(db_name)
- cursor = conn.cursor()
- query = "SELECT * FROM ct_opioiddata_final;"
- opioid_df = pd.read_sql(query, conn)

# Visualizing and Exploring the Data: 
After reading in the data from the database, we began to familiarize ourselves with the various demographic, geographical and binary drug-type data. 

Our first exploration gives us a topline view of **Drug Deaths by Month between 2018-2023** to help us identify any seasonal patterns in drug overdoses in CT. 2021 had the maximum number of incidents, with drug deaths growing by year prior to that. Drug deaths have declined YoY over the last 2 years, but still surpass pre-pandemic levels. Our initial hypothesis was that the winter months would see the highest number of deaths due to seasonal depression, but it was found that peak summer months (May-July) had the highest number of deaths on average.

_can change order, just listing for placeholders_
**visualization 2 description + findings - Marina**

**visualization 3 description + findings - Carleigh**

**visualization 4 description + findings - Kehleboe**


# Cleaning and Transforming the Data for K-Means Clustering:

After exploring the data, we decided to move forward with exploring an unsupervised Machine Learning model to identify clusters and patterns in the data, given that there was no direct output that we would be providing the model.

Prior to initializing the model, we went through multiple steps to "remove noise" and create cohesive data types/values. We initially attempted a K-Means cluster with Zip Codes but got inconclusive data skewed by Zip Codes, so we made the decision to drop those columns. For columns like InjuryLocation where there are 80+ values and multiple values only have 1-2 instances, condensed low-count values into the "Other" bucket. 

As a final step to prepare our data for K-Means clustering, we used the function **pd.get_dummies** to transform string values into their own columns with True/False values. To align with the Drug-Data, we then transformed True = 1 and False = 0. This allows us to easily see the shares of each cluster that are Positive/Negative for each Demographic feature and each Drug Type. 


_steps for K-means cluster - will add later_

# Cluster Characteristics:

**Cluster 1:**

**Cluster 2:**

**Cluster 3:**

**Cluster 4:**

**Cluster 5:**

**Cluster 6:**
- Female
- These individuals are about 75% white, which is a lower share than other clusters. This cluster is more diverse in terms of race, seeing higher than average shares of Black/African American individuals (21%), Asian individuals (1%) and Race Other/Unknown individuals (3%)
- Approximately half of these women were found deceased in their home, versus the other Female cluster (3) who passed away in another Residence
- These women had a higher than average share of deaths from Cocaine (51%) and Benzodiazepines (26%)
- Shares of death attributed to Heroin (12%) and Oxymorphone (0.5%) are lower than other clusters

**Cluster 7:**
- Male
- These men did not die in their homes, but rather other areas such as Hotels/Motels, Outdoors or Other/Unknown
- They tend to be White (84%) or Hispanic (11%)
- Over-indexing compared to other clusters in Ethanol & Fetanyl usage - highest share for these two drug types across clusters at over 89% testing positive for Fetanyl and 33% testing positive for Ethanol
- They gravitate towards Meth/Amphetamine, Hydromorphone, Oxymorphone and Oxycodone at lower rates than other clusters
