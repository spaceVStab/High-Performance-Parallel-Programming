#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <omp.h>
#include <math.h>
#define HI(num) (((num) & 0x0000FF00) << 8) 
#define LO(num) ((num) & 0x000000FF)
typedef struct _PGMData {
    int row;
    int col;
    int max_gray;
    unsigned char *matrix;
} PGMData;


unsigned char *allocate_dynamic_matrix(int row, int col)
{
    unsigned char *ret_val;
    int i;
 
    ret_val = (unsigned char *)malloc(sizeof(unsigned char)*row*col);
    if (ret_val == NULL) 
    {
        perror("memory allocation failure");
        exit(EXIT_FAILURE);
    }
 
    return ret_val;
}
 
void deallocate_dynamic_matrix(unsigned char *matrix)
{
    free(matrix);
}
void SkipComments(FILE *fp)
{
    int ch;
    char line[100];
    while ((ch = fgetc(fp)) != EOF && isspace(ch)) 
    {
        ;
    }
 
    if (ch == '#') 
    {
        fgets(line, sizeof(line), fp);
        SkipComments(fp);
    } 
    else 
    {
        fseek(fp, -1, SEEK_CUR);
    }
} 
void readPGM(const char *file_name, PGMData *data)
{
    FILE *pgmFile;
    char version[3];
    int i, j;
    int lo, hi;
    pgmFile = fopen(file_name, "rb");
    if (pgmFile == NULL)
     {
        perror("cannot open file to read");
        exit(EXIT_FAILURE);
    }
    fgets(version, sizeof(version), pgmFile);
    if (strcmp(version, "P5")) 
    {
        fprintf(stderr, "Wrong file type!\n");
        exit(EXIT_FAILURE);
    }
    SkipComments(pgmFile);
    fscanf(pgmFile, "%d", &data->col);
    SkipComments(pgmFile);
    fscanf(pgmFile, "%d", &data->row);
    SkipComments(pgmFile);
    fscanf(pgmFile, "%d", &data->max_gray);
    fgetc(pgmFile);
 
    data->matrix = allocate_dynamic_matrix(data->row, data->col);
    if (data->max_gray > 255) 
    {
        for (i = 0; i < data->row; ++i) 
        {
            for (j = 0; j < data->col; ++j) 
            {
                hi = fgetc(pgmFile);
                lo = fgetc(pgmFile);
                data->matrix[i*data->col+j] = (hi << 8) + lo;
            }
        }
    }
    else 
    {
        for (i = 0; i < data->row; ++i) {
            for (j = 0; j < data->col; ++j) {
                lo = fgetc(pgmFile);
                data->matrix[i*data->col+j] = lo;
            }
        }
    }
 
    fclose(pgmFile);
    //return data;
 
}
 
/*and for writing*/
 
void writePGM(char *filename, PGMData *data)
{
    FILE *pgmFile;
    int i, j;
    int hi, lo;
 
    pgmFile = fopen(filename, "wb");
    if (pgmFile == NULL) {
        perror("cannot open file to write");
        exit(EXIT_FAILURE);
    }
 
    fprintf(pgmFile, "P5 ");
    fprintf(pgmFile, "%d %d ", data->col, data->row);
    fprintf(pgmFile, "%d ", data->max_gray);
 
    if (data->max_gray > 255) 
    {
        for (i = 0; i < data->row; ++i) 
        {
            for (j = 0; j < data->col; ++j) 
            {
                hi = HI(data->matrix[i*data->col+j]);
                lo = LO(data->matrix[i*data->col+j]);
                fputc(hi, pgmFile);
                fputc(lo, pgmFile);
            }
 
        }
    }
    else 
    {
        for (i = 0; i < data->row; ++i) 
        {
            for (j = 0; j < data->col; ++j) 
            {
                lo = LO(data->matrix[i*data->col+j]);
                fputc(lo, pgmFile);
            }
        }
    }
 
    fclose(pgmFile);
    deallocate_dynamic_matrix(data->matrix);
}



int main()
{

    
    int n_threads = omp_get_max_threads();
  	char c[]="foggy_3.pgm";   //Input file name
  	char o[]="foggy_3_op.pgm";  //Output file name
    PGMData* p=(PGMData*)malloc(sizeof(PGMData));
    PGMData* po=(PGMData*)malloc(sizeof(PGMData));
    readPGM(c,p);
    const int image_width = p->row;
    const int image_height = p->col;
    const int image_size = p->row*p->col;
    const int color_depth = 255;
    unsigned char* output_image = (unsigned char*)malloc(sizeof(unsigned char) * image_size);
    unsigned char* image = p->matrix;
    

    printf("Successful Read\n");

    int* histogram = (int*)calloc(sizeof(int), color_depth);
    double wtime = omp_get_wtime ( );
    #pragma omp parallel for num_threads(n_threads)
    for(int i = 0; i < image_size; i++)
    {
        int image_val = image[i]; 
        #pragma omp critical
        histogram[image_val]++;
    }

    FILE *f = fopen("histogram_ip_3.txt", "wb");
    for(int k=0;k<256;k++)
    	fprintf(f,"%d ",histogram[k]);
    fclose(f);

    float* transfer_function = (float*)calloc(sizeof(float), color_depth);
    
    #pragma omp parallel for num_threads(n_threads) schedule(static,1)
    for(int i = 0; i < color_depth; i++)
    {
        for(int j = 0; j < i+1; j++)
        {
            transfer_function[i] += color_depth*((float)histogram[j])/(image_size);
        }
    }


    #pragma omp parallel for num_threads(n_threads)
    for(int i = 0; i < image_size; i++)
    {
        output_image[i] = transfer_function[image[i]];
    }
    printf("Time: %.6f\n",omp_get_wtime ( )-wtime);
    po->row=p->row;
    po->col=p->col;
    po->max_gray=p->max_gray;
    po->matrix=output_image;
    int* histogramop = (int*)calloc(sizeof(int), color_depth);
    #pragma omp parallel for num_threads(n_threads)
    for(int i = 0; i < image_size; i++)
    {
        int image_val = output_image[i]; 
        #pragma omp critical
        histogramop[image_val]++;
    }
    f = fopen("histogram_op_3.txt", "wb");
    for(int k=0;k<256;k++)
    	fprintf(f,"%d ",histogramop[k]);
    fclose(f);
    writePGM(o,po);
    
    

    printf("Successful Write\n");
    return 0;
}

