from numpy import dot
from numpy import transpose
from numpy import linalg
from numpy import argsort

def PCA(T):
	W=dot(transpose(T),T)
	eig_vals, eig_vecs = linalg.eig(W)
	sort_perm = argsort(eig_vals)[::-1]
	eig_vals.sort()     # <-- This sorts the list in place.
	eig_vecs = eig_vecs[:,sort_perm]
	return eig_vecs
