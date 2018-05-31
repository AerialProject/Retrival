from scipy.spatial import distance
import numpy as np
import time
from sklearn import preprocessing

def dist(database,query):
    # database = preprocessing.normalize(database)
    # query = preprocessing.normalize(query)

    dist = np.zeros([len(database),])
    
    start_time = time.time()
    for i in range(len(database)):
        dist[i] = distance.euclidean(database[i],query)
        # dist[i] = distance.cosine(database[i], query)

    Tdiff = time.time()-start_time
    m, s = divmod(Tdiff, 60)
    h, m = divmod(m, 60)
    print('Time to calculate distances = %dh:%02dm:%0.2fs' % (h,m,s))
    
    return dist

# from sklearn import preprocessing
# np.linalg.norm(preprocessing.normalize(feature))
# tmp = preprocessing.normalize(database_features)
# np.linalg.norm(tmp[-1])
