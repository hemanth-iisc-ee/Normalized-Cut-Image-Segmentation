import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx 
import scipy.sparse as spr
from sklearn.cluster import KMeans 


def preporc_img(img, sigma=5):
	h,w,ch = img.shape[0], img.shape[1], img.shape[2]
	
	lin_img = img.reshape(h*w,ch)
	lin_idx = np.array(range(h*w))

	# right
	tmp = lin_idx + 1
	right = list(zip(lin_idx[tmp%w != 0],tmp[tmp%w != 0]))
	
	# left
	tmp = lin_idx - 1
	left = list(zip(lin_idx[(lin_idx%w != 0) & (tmp >= 0)],tmp[(lin_idx%w != 0) & (tmp >= 0)]))
	
	# down  
	tmp = lin_idx + w
	down = list(zip(lin_idx[tmp < w*h],tmp[tmp < w*h]))
	
	# up  
	tmp = lin_idx - w
	up = list(zip(lin_idx[tmp >= 0],tmp[tmp >= 0]))

	# down right 
	tmp = lin_idx + w +1
	down_right = list(zip(lin_idx[(tmp < w*h) & ((lin_idx+1)%w != 0)],tmp[(tmp < w*h) & ((lin_idx+1)%w != 0)]))

	# up right  
	tmp = lin_idx - w +1
	up_right = list(zip(lin_idx[(tmp >= 0) & ((lin_idx+1)%w != 0)],tmp[(tmp >= 0) & ((lin_idx+1)%w != 0)]))

	# down left 
	tmp = lin_idx + w - 1
	down_left = list(zip(lin_idx[(tmp < w*h) & ((lin_idx)%w != 0)],tmp[(tmp < w*h) & ((lin_idx)%w != 0)]))

	# up right  
	tmp = lin_idx - w - 1
	up_left = list(zip(lin_idx[(tmp >= 0) & ((lin_idx)%w != 0)],tmp[(tmp >= 0) & ((lin_idx)%w != 0)]))

	edges = np.array(right + left + down + up + down_right + down_left + up_right + up_left)

	diff = 0.0
	for i in range(ch):
		diff = diff + (lin_img[edges[:,0],i] - lin_img[edges[:,1],i])**2
	edge_weights = np.exp(-diff/sigma).reshape(len(edges),1)

	nxedges = np.concatenate((np.array(edges),edge_weights),axis=1) 

	G = nx.Graph()
	G.add_weighted_edges_from(nxedges) 

	return G

# Prameters
n_clusters = 4

# Synthetic Image
h,w = 40,60
im = np.zeros((h,w))


im[:h//2,:w//2] = 100
im[h//2:,:] = 200
im[h//2:,w//2:] = 50

noise = 2*np.random.normal(size=im.shape)
img = im + noise
G = preporc_img(img[:,:,np.newaxis])

# # Load an image
# img = cv2.imread('filename')
# G = preporc_img(img[:,:,np.newaxis])


L = nx.normalized_laplacian_matrix(G)
l,v = spr.linalg.eigsh(L, n_clusters ,which='SM')

kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(v)
labels = kmeans.labels_.reshape(h,w)
plt.figure(1)
plt.subplot(2,1,1)
plt.imshow(img,cmap='gray')
plt.title('Input Image')
plt.xticks([])
plt.yticks([])
plt.subplot(2,1,2)
plt.imshow(labels,cmap='jet')
plt.title('Segmented Label Image')
plt.show()


