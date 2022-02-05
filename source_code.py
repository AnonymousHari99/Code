#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies = movies.merge(credits,on='title')


# In[4]:


movies.head()


# In[5]:


movies.info()


# In[6]:


movies.isnull().sum()


# In[7]:


movies.dropna(inplace=True)


# In[8]:


movies.duplicated().sum()


# In[9]:


movies.iloc[0].genres


# In[10]:


def convert(obj):
    L= []
    for i in obj:
        L.append(i['name'])
    return L


# In[11]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[13]:


def convert(obj):
    L= []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[14]:


movies['genres']=movies['genres'].apply(convert)


# In[15]:


movies.head()


# In[16]:


# genres
# keywords
# id
# title
# cast
# crew
#overview

movies=movies[['movie_id','title','overview','keywords','genres','cast','crew']]


# In[20]:


movies.head()


# In[18]:


def convert(obj):
    L= []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[19]:


movies['keywords']=movies['keywords'].apply(convert)


# In[23]:


def convert3(obj):
    L= []
    counter =0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[24]:


movies['cast']=movies['cast'].apply(convert3)


# In[25]:


movies.head()


# In[28]:


def fetch_director(obj):
    L= []
    counter =0
    for i in ast.literal_eval(obj):
        if i['job'] =='Director':
            L.append(i['name'])
            break
    return L


# In[30]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[36]:


movies.head()


# In[32]:


movies['overview'][0]


# In[35]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[37]:


movies.head()


# In[39]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[40]:


movies.head()


# In[41]:


movies['tags'] = movies['overview']+ movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[43]:


movies.head()


# In[44]:


new_df = movies[['movie_id','title','tags']]


# In[45]:


new_df


# In[47]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[51]:


new_df.head()


# In[53]:


new_df['tags'][0]


# In[54]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[55]:


new_df.head()


# In[67]:


get_ipython().system('pip install nltk')


# In[69]:


import nltk


# In[56]:


new_df['tags'][1]


# In[86]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english') 


# In[87]:


cv.fit_transform(new_df['tags']).toarray()


# In[94]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[95]:


vectors[0]


# In[96]:


len(cv.get_feature_names())


# In[97]:


cv.get_feature_names()


# In[70]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[74]:


def stem(text):
    y = []
    for i in text.split():
       y.append(ps.stem(i))
    return " ".join(y)
    


# In[73]:


ps.stem('loving')


# In[75]:


stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron')


# In[77]:


new_df['tags']=new_df['tags'].apply(stem)


# In[78]:


new_df.head()


# In[98]:


cv.get_feature_names()


# In[99]:


from sklearn.metrics.pairwise import cosine_similarity


# In[100]:


cosine_similarity(vectors)


# In[110]:


similarity =cosine_similarity(vectors)


# In[121]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[118]:


sorted(similarity[1], reverse=True)


# In[113]:


new_df['title'] == 'Avatar'


# In[114]:


new_df[new_df['title'] == 'Avatar']


# In[115]:


new_df[new_df['title'] == 'Avatar'].index[0]


# In[116]:


new_df[new_df['title'] == 'Batman Begins'].index[0]


# In[128]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
    return


# In[129]:


recommend('Avatar')


# In[127]:


new_df.iloc[581].title


# In[130]:


recommend('Batman Begins')


# In[ ]:




