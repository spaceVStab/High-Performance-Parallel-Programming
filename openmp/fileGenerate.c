#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void printToFile(char filepath[]){
	
	return;
}

int main(){
	FILE *fptr; 
	// char filepath[100];
	char chartoappend[15];	
	char setChar[4] = {'A','C','T','G'};
	char* filepath = "largeData.txt";
	fptr = fopen(filepath, "wa");


	for(int i=0;i<6400000;i++){
		
		for(int i=0;i<15;i++){
			chartoappend[i] = setChar[(rand() % 4)];
		}
		// chartoappend[15] = '\n';
		fputs(chartoappend, fptr);
		fputs("\n",fptr);
		
	}
	fclose(fptr);
	return 0;

}
