#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

# Read the Excel file and assign it to a DataFrame (you need to specify the file extension, e.g., .xlsx)
df = pd.read_excel('/Users/diogojkv/Downloads/EuropeTop100Attractions_ENG_20190101_20210821.xlsx')

# Display the first few rows of the DataFrame
print(df.head())


# In[4]:


print(df.head())


# In[13]:


most_visited_destinations = df['reviewVisited'].value_counts().head(5)
print(most_visited_destinations)


# In[14]:


average_ratings_by_trip_type = df.groupby('tripType')['reviewRating'].mean()
print(average_ratings_by_trip_type)


# In[15]:


# Check for missing values
missing_data = df.isna().sum()
print(missing_data)

# Drop rows with missing values (if necessary)
df.dropna(inplace=True)

# Impute missing values with a specific value or method (e.g., mean, median)
df['userLocation'].fillna('Unknown', inplace=True)


# In[18]:


from scipy import stats
z_scores = stats.zscore(df['userContributions'])
df = df[(z_scores < 3)]


# In[19]:


# Convert date columns to datetime format
df['extractionDate'] = pd.to_datetime(df['extractionDate'])
df['reviewWritten'] = pd.to_datetime(df['reviewWritten'])
df['reviewVisited'] = pd.to_datetime(df['reviewVisited'])


# In[20]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    # Tokenize the text and remove stopwords
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(filtered_tokens)

df['reviewFullText'] = df['reviewFullText'].apply(preprocess_text)


# In[21]:


# Check for duplicate rows
duplicates = df[df.duplicated()]

# Remove duplicate rows
df.drop_duplicates(inplace=True)


# In[22]:


# Convert userLocation to lowercase and remove leading/trailing spaces
df['userLocation'] = df['userLocation'].str.strip().str.lower()


# In[23]:


# Check for duplicate rows
duplicates = df[df.duplicated()]

# Remove duplicate rows
df.drop_duplicates(inplace=True)


# In[ ]:




