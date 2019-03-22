#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(){
	int N=131072;
	int flag = 1;
	double start_time = omp_get_wtime();
	FILE *file1, *file2;

	file1 = fopen("4_threads_primenum.txt","a");
	file2 = fopen("16_threads_primenum.txt","a");

	#pragma omp parallel num_threads(4)
	{
		#pragma omp for
		for(int i=1;i<=N;i++){
			flag = 1;
			for(int j=2; j <= i/2; j++){

				if((i%j) == 0){
					// divisible by other than 1 and itself thus not prime
					flag = 0;
					break;
				}
			}

			if (flag == 1){
				fprintf(file1, "Prime Number %d\n", i);
				fprintf(file1, "Time spent %f\n", omp_get_wtime() - start_time);
				// printf("%d\n", i);7
				// printf("Time past is %f\n", omp_get_wtime() - start_time);
			}
		}
	}

	start_time = omp_get_wtime();
	#pragma omp parallel num_threads(16)
	{
		#pragma omp for
		for(int i=1;i<=N;i++){
			flag = 1;
			for(int j=2; j <= i/2; j++){

				if((i%j) == 0){
					// divisible by other than 1 and itself thus not prime
					flag = 0;
					break;
				}
			}

			if (flag == 1){
				fprintf(file2, "Prime Number %d\n", i);
				fprintf(file2, "Time spent is %f\n", omp_get_wtime() - start_time);
				// printf("%d\n", i);
				// printf("Time past is %f\n", omp_get_wtime() - start_time);
			}
		}
	}
	return 0;
}
