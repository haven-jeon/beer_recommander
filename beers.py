
# coding: utf-8

# In[8]:


import pandas as pd
from keras.layers import *
from keras.regularizers import *
from keras.models import *
from sklearn.metrics.pairwise import euclidean_distances
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


beers = pd.read_csv('beer_reviews.csv.bz2')
idx_reviewer = beers.groupby('review_profilename').size().reset_index()
idx_reviewer.shape


# In[10]:


user2idx = dict(zip(idx_reviewer.review_profilename, idx_reviewer.index.values))
idx_beers = beers.groupby('beer_name').size().reset_index()
idx_beers.shape


# In[11]:


beer2idx = dict(zip(idx_beers.beer_name, idx_beers.index.values))
idx2beer = dict(zip(idx_beers.index.values,idx_beers.beer_name))

beers['beer_idx'] = [beer2idx.get(b) for b in beers.beer_name]
beers['user_idx'] = [user2idx.get(u) for u in beers.review_profilename]

beers.loc[:, ['user_idx', 'beer_idx', 'review_overall']].as_matrix()

n_users = len(user2idx) 
n_beers = len(beer2idx) 


# In[4]:


user_in = Input(shape=(1,), dtype='int64', name='user_in')
u = Embedding(n_users, 20, input_length=1,embeddings_regularizer=l2(1e-5))(user_in)

beer_in = Input(shape=(1,), dtype='int64', name='beer_in')
b = Embedding(n_beers, 20, input_length=1,embeddings_regularizer=l2(1e-5))(beer_in)

x = Dot(axes=2)([u,b])
x = Flatten()(x)

model = Model([user_in, beer_in], x)

model.compile('Adam',loss='mse')

model.summary()


# In[19]:


from IPython.display import Image
from keras.utils.vis_utils import model_to_dot

Image(model_to_dot(model,show_shapes=True, show_layer_names=False).create(prog='dot', format='png'))


# In[6]:


data_set_mat = beers.loc[:, ['user_idx', 'beer_idx', 'review_overall']].as_matrix()

history = model.fit(x=[data_set_mat[:,0], data_set_mat[:,1]], y=data_set_mat[:,2], 
                    batch_size=64, epochs=10, validation_split=0.1, verbose=2)


# In[12]:


beer_dist = euclidean_distances(X=model.layers[3].get_weights()[0])

#[k for k, v in beer2idx.items() if k.startswith('Guinness')]


# In[13]:


def get_beer_recomm(beer_nm, beer_dist, idx2beer, topN=10):
    q_b_idx = beer2idx[beer_nm]
    beer_dists = beer_dist[q_b_idx]
    orders = np.argsort(beer_dists)
    return(zip(beer_dists[orders[:topN]], [idx2beer[i] for i in orders[:topN]]))
    
    


# In[15]:


rec = get_beer_recomm('Indica India Pale Ale',beer_dist, idx2beer, topN=10)
tuple(rec)


# In[16]:


rec = get_beer_recomm('Samuel Adams Boston Lager',beer_dist, idx2beer, topN=10)
tuple(rec)

