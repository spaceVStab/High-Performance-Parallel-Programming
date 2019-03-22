#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>


float wallX = 200, wallY=100, wallZ=400;

struct Body 
{
	//id starting with 0
	int id;
	double dx,dy,dz;
	double fx,fy,fz;
	//initially all the particles are rest so veliocity whichis at t - del(t)/2 is zero
	double vx,vy,vz;
};

float roundCoord(float var)
{
	int val = (int)(var*100 + 0.5);
	return ((float)val / 100);
}

void calcForce(int p, struct Body particles[1000]){
	// for every particle with p, calc force along x,y,z
	double force_x=0, force_y=0, force_z=0;
	// double force_total[1000][1000][3];
	double dist;
	#pragma omp parallel for private(dist, force_x, force_y, force_z)
	for(int i=0;i<1000;i++){
		if(i==p){
			continue;
		}
		else{
			dist = pow(pow((pow((particles[p].dx - particles[i].dx),2) + pow((particles[p].dy - particles[i].dy),2) + pow((particles[p].dz - particles[i].dz),2)), 1/2),3);
			force_x += ((particles[p].dx - particles[i].dx))/dist;
			force_y += ((particles[p].dy - particles[i].dy))/dist;
			force_z += ((particles[p].dz - particles[i].dz))/dist;
		}
		particles[i].fx = force_x;
		particles[i].fy = force_y;
		particles[i].fz = force_z;
	}
	return;
}

void calcVelocity(int p, struct Body particles[1000], float delTime){
	particles[p].vx += (particles[p].fx * delTime)/2;
	particles[p].vy += (particles[p].fy * delTime)/2;
	particles[p].vz += (particles[p].fz * delTime)/2;

	return;
}


void calcPosition(int p, struct Body particles[1000], float delTime){
	float pos_x, pos_y, pos_z;
	pos_x = particles[p].dx + particles[p].vx * delTime;
	pos_y = particles[p].dy + particles[p].vy * delTime;
	pos_z = particles[p].dz + particles[p].vz * delTime;


	// see for the wall
	// if position exceed then only vel changes
	if(pos_x > wallX){
		pos_x = 2*wallX - pos_x;
		particles[p].vx = (-1) * particles[p].vx;
	}
	if(pos_x < 0) {pos_x = 0 - pos_x; particles[p].vx = (-1) * particles[p].vx;}
	if(pos_y > wallY) {pos_y = 2*wallY - pos_y; particles[p].vy = (-1) * particles[p].vy;}
	if(pos_y < 0) {pos_y = 0 - pos_y; particles[p].vy = (-1) * particles[p].vy;}
	if(pos_z > wallZ) {pos_z = 2*wallZ - pos_z; particles[p].vz = (-1) * particles[p].vz;}
	if(pos_z < 0) {pos_z = 0 - pos_z; particles[p].vz = (-1) * particles[p].vz;}
	particles[p].dx = pos_x;
	particles[p].dy = pos_y;
	particles[p].dz = pos_z;

	return;
}

void savePositions(struct Body particles[1000]){
	FILE *fptr;
	fptr = fopen("FinalPositions_8.txt","a");
	char buffer[100];
	for (int i=0;i<1000;i++){
		sprintf(buffer, "%.2f %.2f %.2f\n", particles[i].dx, particles[i].dy, particles[i].dz);
		fputs(buffer, fptr);
	}
	fputs("\n", fptr);
	fclose(fptr);
	printf("After 100 time steps, the coordinates are saved.\n");
	return;
}

void saveLog(int t_step, double time){
	FILE *fptr;	
	char buffer[100];
	fptr = fopen("LOG_8.txt","a");
	sprintf(buffer, "%d %.2lf\n", t_step, time);
	fputs(buffer, fptr);
	fclose(fptr);
	return;
}

void initForce(struct Body particles[1000]){
	#pragma omp parallel for 
	for(int i=0;i<1000;i++){
		particles[i].fx = 0;
		particles[i].fy = 0;
		particles[i].fz = 0;
	}
	return;
}



int main(){
	char c[1000];
	char * coordinates;
	float myvar;
	FILE *fptr;
	int SIM_STEPS = 720000;
	float SIM_TIME = (float)SIM_STEPS * 0.01;

	fptr = fopen("Traj.txt","r");

	struct Body particles[1000];
	int wallX = 200, wallY=100, wallZ=400;
	// read the initial position for the coordinates
	printf("Reading the coordinates\n");
	for(int i=0;i<1008;i++){
		if (i<8){
			char buffer[100];
			fgets(buffer, 100, fptr);
			continue;
		}
		particles[i-8].id = i-8;	
		fscanf(fptr, "%f", &myvar);
		particles[i-8].dx = roundCoord(myvar);

		fscanf(fptr, "%f", &myvar);
		particles[i-8].dy = roundCoord(myvar);
		
		fscanf(fptr, "%f", &myvar);
		particles[i-8].dz = roundCoord(myvar);
	}

	fclose(fptr);

	omp_set_num_threads(8);

	// do it serially first
	printf("The coordinates have been read\n");
	// init all forces to zero 
	#pragma omp parallel for 
	for(int i=0;i<1000;i++){
		particles[i].vx = 0.0;
		particles[i].vy = 0.0;
		particles[i].vz = 0.0;
	}

	float delTime;
	float time=0;
	int t_step = 0;
	double start = omp_get_wtime();
	for(float time=0;time<SIM_TIME;time+=0.01){
		initForce(particles);
		//printf("Force initiated\n");
		t_step+=1;
		#pragma omp parallel for
		for(int p=0;p<1000;p++){
			
			calcForce(p, particles);
			// #pragma omp barrier
			// printf("Force calculated\n");
			if (time==0){
				calcVelocity(p, particles, 0.01/2);
				// #pragma omp barrier
				calcPosition(p, particles, 0.01/2);
				// #pragma omp barrier
			}
			else{
				calcVelocity(p, particles, 0.01);
				// #pragma omp barrier
				calcPosition(p, particles, 0.01);
				// #pragma omp barrier
			}
			// calcVelocity(p, particles, delTime);
			// after calculating the half step velocity calculate the position of the particles

			// calcPosition(p, particles, 0.01);
			// after the n+1 th position is calculated, calculate the full step velcoity
		}
		if((t_step%100) == 0){
			printf("%d\n", t_step);
			savePositions(particles);
		}
		double nowTime = omp_get_wtime() - start;
		if(t_step < 100){
			saveLog(t_step, nowTime);
		}

	}
	printf("%.2lf\n", omp_get_wtime()- start);
	return 0;
}
