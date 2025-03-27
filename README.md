# Inside Connecticut's Opioid Crisis: A Deep Dive into Drug-Related Deaths Using Unsupervised Machine Learning
Project 4 Penn Data Bootcamp | Jillian Walter, Carleigh West, Marina Connolly, Kehleboe Gongloe

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

Our second visualization shows the **distribution of incidents by race and sex**. The chart clearly shows that males have significantly higher incident counts across all racial categories compared to females. White individuals have the highest number of incidents overall, followed by Black or African American and Hispanic. This visualization helps understand the demographic patterns in the opioid data across Connecticut.

Our third exploration looks at **Total Deaths by Drug Type between 2018-2023 in Connecticut** which helped us to determine the drugs responsible for the most deaths. Fentanyl, Cocaine, and Ethanol (alcohol) were the leading culprits in this data set, followed closely by Benzodiazepines and Heroin. I think it is important to recognize, whe exploring this part of the data, that the number of fentanyl deaths being the highest corraborates the facts and statistics about Fentanyl and its position in the opioid epidemic. A Time article from September 2024 reports that fentanyl-related overdose deaths increased from 3,000 in 2013 to over 71,000 in 2022, underscoring the dramatic rise in fentanyl's impact.

For the next visualization we wanted to explore a **map plotting the deaths from 2018-2023 throughout Connecticut**. We then divided our map to show each year of data we had available. 
We decided to use Tableau for this analysis as we wanted to make it dynamic.
Starting in 2018, we show the number of deaths color-separated by male and female. Then we created a Parameter that depicts death per drug, making the map dynamic. We also created a layer to show population density per Zip Code. The Map shows that the Mid-South corridor has been the most affected throughout the years. Even for the least-found drugs.


# Cleaning and Transforming the Data for K-Means Clustering:

After exploring the data, we decided to move forward with exploring an unsupervised Machine Learning model to identify clusters and patterns in the data, given that there was no direct output that we would be providing the model.

Prior to initializing the model, we went through multiple steps to "remove noise" and create cohesive data types/values. We initially attempted a K-Means cluster with Zip Codes but got inconclusive data skewed by Zip Codes, so we made the decision to drop those columns. For columns like InjuryLocation where there are 80+ values and multiple values only have 1-2 instances, condensed low-count values into the "Other" bucket. 

As a final step to prepare our data for K-Means clustering, we used the function **pd.get_dummies** to transform string values into their own columns with True/False values. To align with the Drug-Data, we then transformed True = 1 and False = 0. This allows us to easily see the shares of each cluster that are Positive/Negative for each Demographic feature and each Drug Type. 

# Elbow Method and K-Means Cluster Model Creation:

To perform the Cluster Analysis, we need to start by determining the optimal number of clusters. Using the Elbow Method to calculate inertia at each of the cluster numbers in the For Loop, we found 7 clusters to be the optimal number of clusters for the 7.5k data records. 
![image](https://github.com/user-attachments/assets/abeb8dd7-664b-439e-a665-c334a68cb4ca)

We then work to define the K-Means model with 7 clusters and a random state of 3 for reproducability. This model is then fit and predict the Opioid dummy data created above. Once the model is fitted and predictions are made, we create an additional cluster column for each record in the dataframe, to assign the record to its respective cluster. 
For analysis, given the volume of columns, we start by looking at the demographic columns (Sex, Race, Place of Death). We create a new dataframe called cluster_averages_demographics, using the **groupby()** and **mean()** functions to get the average value of each cluster for each column. Because all values in the dataframe are 0 or 1, the average represents the _share_ of individuals in that dataframe that are positive/True for each of the columns. We used the **style.format()** function to change the format of the averages to percentages for easier readability. The same process is repeated for a second cluster_averages_drugtypes, which looks at the share of each cluster that tested positive for each drug. Screenshot examples of these tables used in analysis are below:

![image](https://github.com/user-attachments/assets/46634577-83ad-44ee-839a-93b82bbeca38)
![image](https://github.com/user-attachments/assets/23f5846f-b944-44f7-b8cd-1ae10975a99b)

# Cluster Characteristics:

**Summary:**
Fentanyl was overall found in at least 72% (up to 89%) of deaths in each cluster, indicating that it is a primary factor of death. Incidentally, cocaine is found at the second highest rate across clusters, aligning with the hypothesis that many Cocaine deaths were due to the drug being laced with Fetanyl. 85% of Cocaine deaths also had Fetanyl present, with similar patterns across other drug types.  Given the amount of drugs in each individual (average 2.4 drugs per individuals) it can be hypothesised that users were not knowingly buying drugs that were pure (were likely laced with other drugs). According to the CDC, Fetanyl is extremely potent: 50x stronger than Heroin and 100x stronger than Morphine. 

**Cluster 1:**
- More than 99% of the individuals in this cluster are men and a 100% of them are Black
- Only 23% of the deceased where found home, while almost 30% of them were found in other Residences/Houses with other lower percentages for other places.
- Cocaine and Fentanyl were most commonly found in their bodies (55 and 86%, respectively), and it's a pretty close number when compared to the second cluster that we found, which varies more in race.

**Cluster 2:**
- This is another cluster composed of more than 99% men, but on this one most of them are White, and White (Hispanic)
- 100% of deceased in this cluster were found in someone else's Residence.
- This is the only cluster where Benzodiazepine was not found in any of the deceased. As with all the other clusters, Fentanyl was found at the highest rate (84%), followed by Cocaine, Ethanol and Heroin.
- Compared to the other clusters, this is the lowest in Oxycodone (5%), and very low rates of Oxymorphone, Hydrocodone, and Hydromorphone

**Cluster 3:**
- This cluster predominantly consists of white males (the highest Race-White share across groups at 96.5%) who passed away in their own homes (100% of deaths)
- This cluster over-indexed in Fentanyl and Oxycodone, ranking second highest across clusters in Fentanyl share of deaths (87%). Oxycodone rates were lower than other drug types at 15%, but they saw higher rates of this drug than any other cluster.
- Morphine rates were lowest in this cluster compared to other clusters.

**Cluster 4:**
- This cluster is one of two female clusters, with individuals across races but overall less racially diverse than cluster  6, with nearly all individuals being White (81%) or Hispanic (6%)
- 100% of these women were found deceased in another's home, while the other Female cluster saw mixtures of death locations, though primarily in their own home
- This cluster saw lower than average rates of the more popular drugs across other clusters, such as Heroin, Cocaine and Fentanyl (ranking lowest/second lowest in terms of shares across those drugs)
- Rather, these individuals saw higher than average shares (ranking top or second) for less popular drugs such as Hydrocodone, Benzodiazepines, Methadone, Tramadol and Hydromorphone

**Cluster 5:**
- This cluster was similar to cluster 4, but Male. This group was also less racially diverse than other groups, seeing 94% of the individuals being White and 4% being Hispanic. These individuals were largely found in another's home, similar to cluster 4.
- 100% of these individuals were found with Benzodiazepines, the highest across any cluster. This type of drug is used to treat Anxiety Disorders. Methamphetamine is also higher than any other cluster within this cluster, which is used to treat ADHD. This can indicate that those in this cluster were abusing prescription drugs, clinically used to treat ADHD + Anxiety, for sedative effects.

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

**Sources:**
- https://nypost.com/2025/03/19/us-news/revolutionary-nyc-program-for-addicts-has-fatal-overdoses-plummeting/
- https://time.com/7023415/the-truth-about-fentanyl-essay/
- https://www.cdc.gov/overdose-prevention/about/understanding-the-opioid-overdose-epidemic.html
