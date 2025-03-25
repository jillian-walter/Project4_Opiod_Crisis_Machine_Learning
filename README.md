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

  db_name = "opioid_analysis.sqlite"
  
  conn = sqlite3.connect(db_name)
  
  cursor = conn.cursor()

  query = "SELECT * FROM ct_opioiddata_final;"
  
  opioid_df = pd.read_sql(query, conn)

