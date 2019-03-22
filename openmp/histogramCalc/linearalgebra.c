#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main()
{
	double A[15][15], b[15];
	double N=15;
	int i=0,j=0,k=0;
	time_t t;
	srand(6);
	
	printf("\nTaking 15 x 15 matrix\n");
	for(i=0;i<N;i++)
	{
		b[i]=rand()%15;
		for (j=0; j<N;j++)
		{
			A[i][j]=rand()%15;
			printf("%.0f ", A[i][j]);
		}
		printf("= %.0f \n", b[i]);
	}

		for (i = 0; i<15-1; ++i)
		{
			#pragma omp parallel for private(j,k)
			for (j = i+1; j<15; ++j)
			{
				double ratio = A[j][i]/A[i][i];
				b[j]-= ratio * b[i];
				for (k = i; k<15; ++k)
				{
					A[j][k]-= ratio * A[i][k];
				}
			}
		}
		for(int i=0;i<N;i++)
		{
			for (int j=0; j<N;j++)
			{
				printf("%.2f \t ", A[i][j]);
			}	
		printf("= %.3f \n", b[i]);
		}
	
	return 0;
}