#pragma once 

#include <complex>

#include <gsl/gsl_sf.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_linalg.h>

class CGSLMatrix_Complex{
	
    public:
		int dim;
		void SolveLinearEqs(std::complex<double> *y, std::complex<double> **A, std::complex<double> *x);
		void EigenFind(std::complex<double> **A, std::complex<double> **eigenvec, double *eigenval); // A must be Hermittian
		void Invert(std::complex<double> **A, std::complex<double> **Ainv); // A must  be Herm.
		CGSLMatrix_Complex(int dimset);
		~CGSLMatrix_Complex();
	
    private:
		gsl_vector *eval;
		gsl_matrix_complex *evec;
		gsl_vector_complex *g;
		gsl_eigen_hermv_workspace *w;
		gsl_permutation *p;
		std::complex<double> **U;
		gsl_matrix_complex *m;
		gsl_vector_complex *v;
	};

CGSLMatrix_Complex::CGSLMatrix_Complex(int dimset){
	dim=dimset;
	eval=NULL;
	evec=NULL;
	w=NULL;
	g=NULL;
	p=NULL;
	U=NULL;
	m=NULL;
	v=NULL;
}

CGSLMatrix_Complex::~CGSLMatrix_Complex(){
	if(eval!=NULL) gsl_vector_free(eval);
	if(evec!=NULL) gsl_matrix_complex_free(evec);
	if(w!=NULL) gsl_eigen_hermv_free(w);
	if(g!=NULL) gsl_vector_complex_free(g);
	if(p!=NULL) gsl_permutation_free(p);
	if(m!=NULL) gsl_matrix_complex_free(m);
	if(v!=NULL) gsl_vector_complex_free(v);
	if(U!=NULL){
		for(int i=0;i<dim;i++)
			delete [] U[i];
		delete [] U;
	}
}

void CGSLMatrix_Complex::EigenFind(std::complex<double> **A,std::complex<double> **eigenvec,double *eigenval){
	std::complex<double> ci(0.0,1.0);
	gsl_complex z;
	//gsl_matrix_complex *m=gsl_matrix_complex_alloc(dim,dim);
	if(m==NULL) m=gsl_matrix_complex_alloc(dim,dim);
	int i,j;
	for(i=0;i<dim;i++){
		for(j=0;j<dim;j++){
			GSL_SET_COMPLEX(&z,real(A[i][j]),imag(A[i][j]));
			gsl_matrix_complex_set(m,i,j,z);
		}
	}

	if(eval==NULL) eval=gsl_vector_alloc(dim);
	if(evec==NULL) evec=gsl_matrix_complex_alloc(dim,dim);
	if(w==NULL) w=gsl_eigen_hermv_alloc(dim);

	gsl_eigen_hermv(m,eval,evec,w);

	//gsl_eigen_hermv_sort(eval,evec,GSL_EIGEN_SORT_ABS_ASC);

	for(i=0;i<dim;i++){
		eigenval[i]=gsl_vector_get(eval,i);
		for(j=0;j<dim;j++){
			z=gsl_matrix_complex_get(evec,i,j);
			eigenvec[i][j]=GSL_REAL(z)+ci*GSL_IMAG(z);
		}
	}

}

void CGSLMatrix_Complex::Invert(std::complex<double> **A,std::complex<double> **Ainv){
	std::complex<double> ci(0.0,1.0);
	gsl_complex z;
	//gsl_matrix_complex *m=gsl_matrix_complex_alloc(dim,dim);
	if(m==NULL) m=gsl_matrix_complex_alloc(dim,dim);
	int i,j;
	for(i=0;i<dim;i++){
		for(j=0;j<dim;j++){
			GSL_SET_COMPLEX(&z,real(A[i][j]),imag(A[i][j]));
			gsl_matrix_complex_set(m,i,j,z);
		}
	}

	if(eval==NULL) eval=gsl_vector_alloc(dim);
	if(evec==NULL) evec=gsl_matrix_complex_alloc(dim,dim);
	if(w==NULL) w=gsl_eigen_hermv_alloc(dim);

	gsl_eigen_hermv(m,eval,evec,w);
	//gsl_eigen_hermv_sort(eval,evec,GSL_EIGEN_SORT_ABS_ASC);

	if(U==NULL){
		U=new std::complex<double> *[dim];
		for(i=0;i<dim;i++) U[i]=new std::complex<double>[dim];
	}

for(i=0;i<dim;i++){
	for(j=0;j<dim;j++) {
		z=gsl_matrix_complex_get(evec,j,i);
		U[i][j]=GSL_REAL(z)+ci*GSL_IMAG(z);
		Ainv[i][j]=0.0;
	}
}
int k;
for(i=0;i<dim;i++){
	for(j=0;j<dim;j++){
		for(k=0;k<dim;k++)
			Ainv[i][j]+=U[k][i]*conj(U[k][j])/gsl_vector_get(eval,k);
	}
}
}

void CGSLMatrix_Complex::SolveLinearEqs(std::complex<double> *y,std::complex<double> **A,std::complex<double> *x){
	std::complex<double> ci(0.0,1.0);
	int i,j,s;
	gsl_complex z;

	if(m==NULL) m=gsl_matrix_complex_alloc(dim,dim);
	if(v==NULL) v=gsl_vector_complex_alloc(dim);
	for(i=0;i<dim;i++){
		GSL_SET_COMPLEX(&z,real(y[i]),imag(y[i]));
		gsl_vector_complex_set(v,i,z);
		for(j=0;j<dim;j++){
			GSL_SET_COMPLEX(&z,real(A[i][j]),imag(A[i][j]));
			gsl_matrix_complex_set(m,i,j,z);
		}
	}

	if(g==NULL) g = gsl_vector_complex_alloc (dim);
	if(p==NULL) p = gsl_permutation_alloc (dim);


	gsl_linalg_complex_LU_decomp (m, p, &s);
	gsl_linalg_complex_LU_solve (m, p, v, g);

	for(i=0;i<dim;i++){
		z=gsl_vector_complex_get(g,i);
		x[i]=GSL_REAL(z)+ci*GSL_IMAG(z);
	}
}
