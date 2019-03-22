#include <stdio.h>
#include <omp.h>

int main(){
	int N=131702;
	int flag = 1;
	double start_time = omp_get_wtime();

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
			printf("%d\n", i);
			printf("Time past is %f\n", omp_get_wtime() - start_time);
		}
	}
	return 0;
}
