def pca(matrix,dimension):
	""" performs PCA on matrix
		dimension is the dimension of the projection
		returns a projection into a lower dimension embedding"""
	Mean=mean(matrix,axis=0);
	M=reshape(Mean,data.shape[0],1);
	C=matrix-M;
	W=dot(transpose(C),C);
	S,U= linalg.eig(W);

	perm=argsort(S)[::-1];
	U=U[:,perm];
	return dot(C,U[:,:dimension]);