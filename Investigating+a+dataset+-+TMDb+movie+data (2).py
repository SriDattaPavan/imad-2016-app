
# coding: utf-8

# # Investigating TMDb movie dataset

# <a id='intro'></a>
# ## Introduction
# 
# 
# >### The TMDb dataset is collected over different movies over the years.
# >>This data set contains information about 10,000 movies collected from The Movie Database (TMDb), including user ratings and revenue.
# >> This data helps us to analyse the movie trends and answer some related questions
# >## Questions to be answered:
# >#### Question1: How are runtime, popularity, revenue and frequency of movies varying over the years?
# >#### Question2:How does genre effect a movie?

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('pylab inline')
import seaborn as sns


# ### Data Wrangling Phase:

# In[3]:


#reading the data
tmdb_data = pd.read_csv('data/tmdb_5000_movies.csv')
tmdb_data.head()


# In[4]:


tmdb_data.describe()


# In[6]:


mean_data = tmdb_data.mean(skipna = True)


# In[7]:


# Here the data is filtered according to the conditions provided

tmdb_data['budget'] = tmdb_data.budget.mask(tmdb_data.budget < 100, mean_data.budget)
tmdb_data['revenue'] = tmdb_data.revenue.mask(tmdb_data.revenue < 100, mean_data.revenue)
tmdb_data['runtime'] = tmdb_data.runtime.mask(tmdb_data.runtime < 5, mean_data.runtime)


# In[8]:


tmdb_data.describe()


# In[9]:


# Check how many elements of each column are null
tmdb_data.isnull().sum()


# ### Data cleaning:

# In[10]:


tmdb_data.head()


# In[11]:


#Remove unnecessary columns
tmdb_data = tmdb_data.drop(['keywords','status','spoken_languages','homepage','tagline'], axis=1)


# ### Editing column names:

# In[12]:


#Obtain the year from the given date

from datetime import datetime as d
def obt_year(data):
    '''Here we take a date as input and 
    return the year. If date is not
    mentioned, 2009 is returned
    '''
    if str(data)!='nan':
        return d.strftime(d.strptime(str(data), "%Y-%m-%d"), "%Y")
    else:
        return "2009"
#All the years are added to a new column
tmdb_data["release_year"] = tmdb_data["release_date"].apply(obt_year)


# In[13]:


tmdb_data.head()


# In[14]:


#Obtain the genre of the film
def split_genres(value):
    '''The genres given are in a very confusing
    format. So, this function takes each genre
    as an input and splits it and takes the first mentioned genre
    '''
    str=""
    lis = value.split(":")
    for i in range(0,len(lis)):
        if "name" in lis[i]:
            lis1 = lis[i+1].split("}")[0][2:-1]
            str=str+"||"+lis1
    return str[2:].split("||")[0]
#All the genres are added in a new column
tmdb_data['genre'] = tmdb_data['genres'].apply(split_genres)
tmdb_data.head()


# ## Exploring the data:

# ### Question1: How are runtime, popularity, revenue and frequency of movies releasing in an year varying over the years?

# #### Variation in frequency of movies releasing in an year:

# In[16]:


years_grouped_data = tmdb_data.groupby("release_year")
years_mean_data = tmdb_data.groupby("release_year").mean()


# In[17]:


#Plot the trend of frequency against release_year
years_grouped_data.describe()['budget']['count'].plot( title='Frequency of movies in each year')
plt.ylabel('Frequency')


# ### Inference:
# ##### From the above plotted histogram, it can be observed that frequency of movies released increased drastically from the year after 1990

# #### Variation in popularity:

# In[18]:


#Plot the trend of popularity against release_year
years_mean_data['popularity'].plot(title='Popularity of movies over the years')
plt.ylabel("Popularity")


# In[90]:


years_mean_data['popularity'].describe()


# In[19]:


years_mean_data['popularity'].hist()
plt.title("Popularity of movies")
plt.xlabel("Popularity")
plt.ylabel("Number of movies")


# ### Inference:

# ##### Considering the mean, we can say that popularity for movies was less before 1958. But after 1958, even though there are notable variations, the trend went increasing. Popularity slowly increased. Popularity is high for the movies released around 1970. After that the popularity is maintained at a mean value and again started increasing consideraably after 2008.

# #### Variation in runtime:

# In[20]:


#Plot the trend of runtime against release_year
years_mean_data['runtime'].plot(title='Runtime over the years')
plt.ylabel('runtime')


# In[22]:


years_mean_data['runtime'].hist()
plt.title('Runtime')
plt.xlabel("runtime")
plt.ylabel("Number of movies")


# In[23]:


years_mean_data['runtime'].describe()


# ##### The runtime is roughly the same over the years. It did not vary much even though we can see small variations in the graph.
# ##### The distribution is right skewed.

# In[25]:


#Plot the trend of revenue against release_year
years_mean_data['revenue'].plot(title="Revenue of movies over the years")
plt.ylabel("Revenue")


# ###### We can clearly see that the mean revenue for movies has been increasing gradually. Even here we have some ups and downs, but, considering periods of years, we can see the increase in revenue.

# ### Question2: How does genre effect a movie?

# #### Influence of genre:

# In[26]:


tmdb_data.head()


# In[27]:


#Make a copy of the previous data
genre_copy_data = tmdb_data.copy()


# In[28]:


genre_copy_data.head()


# In[29]:


genre_copy_data['genre'].describe()


# In[30]:


g = genre_copy_data.groupby('genre').mean()
y = genre_copy_data.groupby('release_year').mean()


# In[31]:


g['popularity'].plot(kind='bar')
plt.ylabel("Popularity")


# ###### Clearly science fiction has a lot of popularity which is followed by adventure films.

# In[32]:


g['revenue'].plot(kind='bar')
plt.title("Revenue for genres")
plt.ylabel("revenue")


# #### Inference:

# ##### As we can observe, Animation and Action genres earn the highest revenue.
# ### Thus we can interpret that genre influences the revenue and popularity for a movie significantly.

# ## Conclusions:
# <p>For the first question posed(i.e How are frequency, popularity, revenue and runtime varying over the years?), we observed from the explorating phase, the following:</p>
# <p><li><ul>Frequency of movies released increased drastically from the year after 1990</ul>
# <ul>Popularity for movies was less before 1958. But after 1958, even though there are notable variations, the trend went increasing. Popularity slowly increased. Popularity is high for the movies released around 1970. After that the popularity is maintained at a mean value and again started increasing consideraably after 2008.</ul>
# <ul>The runtime is roughly the same over the years. It did not vary much.</ul>
# <ul>Mean revenue for movies has been increasing gradually. Even here we have some ups and downs, but, considering periods of years, we can see the increase in revenue.</ul>
# </li></p>
# <p>For the second question posed(i.e How does genre effect a movie?), we observed that genre effects both the popularity and revenue for a movie. Clearly, Science fiction has high popularity followed by adventure and revenue wise, animation is in the leading followed by adventure films. But causation does not imply correlation. We need more data here like the directors and cast details to conduct the observation</p>

# ### Limitations:

# <p>
# <li>
# <ul>Some of the data is incomplete. So, the interpretation can be done a bit more precisely</ul>
# <ul>More information is required. Directors and lead cast can also influence popularity and revenue for a movie.</ul>
# </li>
# </p>

# <h5>References:</h5>
# <p><a href="https://stackoverflow.com/">Stackoverflow</a> was used for many coding related queries</p>
# <p><a href="https://matplotlib.org">Mathplotlib</a> was used</p>
# <p><a href="http://pandas.pydata.org/pandas-docs/stable/">Pandas documentation</a> was used</p>
