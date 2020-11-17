#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/// DATA TYPE
typedef struct
{
    unsigned char b;
    unsigned char g;
    unsigned char r;

} Pixel;

typedef struct
{
    unsigned short type;
    unsigned int size;              // File size in bytes
    unsigned short reserved1;
    unsigned short reserved2;
    unsigned int offset;            // Offset to image data in bytes from beginning of file
    unsigned int dib_header_size;
    int  width_px;          // Width of the image
    int  height_px;         // Height of image
    unsigned short num_planes;        // Number of color planes
    unsigned short bits_per_pixel;    // Bits per pixel
    unsigned int compression;       // Compression type
    unsigned int image_size_bytes;  // Image size in bytes
    int  x_resolution_ppm;  // Pixels per meter
    int  y_resolution_ppm;  // Pixels per meter
    unsigned int num_colors;        // Number of colors
    unsigned int important_colors;  // Important colors
} BMPHeader;

typedef struct
{
    BMPHeader header;
    Pixel* pixels;
} Image;
typedef struct
{
    unsigned char height, width, area;
} Template;

/// FUNCTIONS





/// 1: metoda XORSHIFT32

unsigned int XorShift32(unsigned int *seed)
{
    unsigned int x = *seed;
    x = x ^ x << 13;
    x = x ^ x >> 17;
    x = x ^ x << 5;
    *seed = x;

    return x;
}





/// 2: UPLOAD A BMP IMAGE TO INTERN MEMORY IN LINEARIZED FORM
Image* UploadImageToMemory(char* source_file)
{
    printf("\n");
    printf("We read the image %s in memory\n", source_file);

    FILE *fin;
    fin = fopen(source_file, "rb");

    if(fin == NULL)
    {
        printf("We didn't find the image!");
        exit(EXIT_FAILURE);
    }

    Image *img;
    img = (Image*) calloc(1, sizeof(Image));

    fseek(fin, 0, SEEK_SET);
    fread(&img->header.type, sizeof(img->header.type), 1, fin);
    fread(&img->header.size, sizeof(img->header.size), 1, fin);
    fread(&img->header.reserved1, sizeof(img->header.reserved1), 1, fin);
    fread(&img->header.reserved2, sizeof(img->header.reserved2), 1, fin);
    fread(&img->header.offset, sizeof(img->header.offset), 1, fin);
    fread(&img->header.dib_header_size, sizeof(img->header.dib_header_size), 1, fin);
    fread(&img->header.width_px, sizeof(img->header.width_px), 1, fin);
    fread(&img->header.height_px, sizeof(img->header.height_px), 1, fin);
    fread(&img->header.num_planes, sizeof(img->header.num_planes), 1, fin);
    fread(&img->header.bits_per_pixel, sizeof(img->header.bits_per_pixel), 1, fin);
    fread(&img->header.compression, sizeof(img->header.compression), 1, fin);
    fread(&img->header.image_size_bytes, sizeof(img->header.image_size_bytes), 1, fin);
    fread(&img->header.x_resolution_ppm, sizeof(img->header.x_resolution_ppm), 1, fin);
    fread(&img->header.y_resolution_ppm, sizeof(img->header.y_resolution_ppm), 1, fin);
    fread(&img->header.num_colors, sizeof(img->header.num_colors), 1, fin);
    fread(&img->header.important_colors, sizeof(img->header.important_colors), 1, fin);

    int array_dimension = img->header.width_px * img->header.height_px;
    img->pixels = (Pixel*) calloc(array_dimension, sizeof(Pixel)); //allocate memory for Image and initialize with 0

    fread(img->pixels, array_dimension, sizeof(Pixel), fin);
     //calculate the padding for a single line
    int padding = 0;
    if(img->header.width_px % 4 != 0)
    {
        padding = 4 - (3 * img->header.width_px) % 4;
    }

    int i,a;
    for(i = img->header.height_px-1; i>=0 ; i--)
    {
        a = i * img->header.width_px;
        fread(&img->pixels[a], sizeof(Pixel), img->header.width_px, fin);//we read a line without padding
        fseek(fin, padding, SEEK_CUR);
    }

    /*printf("type: %u\n", img->header.type);
    printf("size: %u\n", img->header.size);
    printf("reserved1: %u\n", img->header.reserved1);
    printf("reserved1: %u\n", img->header.reserved2);
    printf("offset: %u\n", img->header.offset);
    printf("dib_header_size: %u\n", img->header.dib_header_size);
    printf("width_px: %d\n", img->header.width_px);
    printf("height_px: %d\n", img->header.height_px);
    printf("num_planes: %u\n", img->header.num_planes);
    printf("bits_per_pixel: %u\n", img->header.bits_per_pixel);
    printf("compression: %u\n", img->header.compression);
    printf("image_size_bytes: %u\n", img->header.image_size_bytes);
    printf("x_resolution_ppm: %d\n", img->header.x_resolution_ppm);
    printf("y_resolution_ppm: %d\n", img->header.y_resolution_ppm);
    printf("num_colors: %d\n", img->header.num_colors);
    printf("important_colors: %d\n", img->header.important_colors);*/

    printf("\n");

    return img;
}





/// 3: SAVE A BMP IMAGE(STOCK IN LINEARIZED FORM IN INTERN MEMORY) TO EXTERN MEMORY

void WriteImageOnDisk(Image* img, char* destination_file)
{
    printf("\n");
    printf("We save the image %s from memory to disk.\n", destination_file);

    FILE *fout;
    fout = fopen(destination_file, "wb+");

    fwrite(&img->header.type, 1, sizeof(img->header.type), fout);
    fwrite(&img->header.size, 1, sizeof(img->header.size), fout);
    fwrite(&img->header.reserved1, 1, sizeof(img->header.reserved1), fout);
    fwrite(&img->header.reserved2, 1, sizeof(img->header.reserved2), fout);
    fwrite(&img->header.offset, 1, sizeof(img->header.offset), fout);
    fwrite(&img->header.dib_header_size, 1, sizeof(img->header.dib_header_size), fout);
    fwrite(&img->header.width_px, 1, sizeof(img->header.width_px), fout);
    fwrite(&img->header.height_px, 1, sizeof(img->header.height_px), fout);
    fwrite(&img->header.num_planes, 1, sizeof(img->header.num_planes), fout);
    fwrite(&img->header.bits_per_pixel, 1, sizeof(img->header.bits_per_pixel), fout);
    fwrite(&img->header.compression, 1, sizeof(img->header.compression), fout);
    fwrite(&img->header.image_size_bytes, 1, sizeof(img->header.image_size_bytes), fout);
    fwrite(&img->header.x_resolution_ppm, 1, sizeof(img->header.x_resolution_ppm), fout);
    fwrite(&img->header.y_resolution_ppm, 1, sizeof(img->header.y_resolution_ppm), fout);
    fwrite(&img->header.num_colors, 1, sizeof(img->header.num_colors), fout);
    fwrite(&img->header.important_colors, 1, sizeof(img->header.important_colors), fout);

    //calculate the padding for a single line
    int padding = 0;

    if(img->header.width_px % 4 != 0)
    {
        padding = 4 - (3 * img->header.width_px) % 4;
    }

    int i, j, a;
    for(i = 0; i < img->header.height_px; i++)
    {
        a = i * img->header.width_px;
        fwrite(&img->pixels[a], sizeof(Pixel), img->header.width_px, fout);//write a line without padding

        for(j = 0; j < padding; j++)
        {
            unsigned char paddingByte = 0;
            fwrite(&paddingByte, sizeof(unsigned char), 1, fout);//write the padding bytes(null)
        }
    }


    fflush(fout);
    fclose(fout);

    printf("\n");
}





/// 4: ENCRYPT A BMP IMAGE

Image* CloneImage(Image* img)
{
    Image* imgResult = (Image*) calloc(1, sizeof(Image));
    int array_dimension = img->header.width_px * img->header.height_px;
    memcpy(imgResult, img, sizeof(Image));

    imgResult->pixels = NULL;
    imgResult->pixels = (Pixel*) calloc(array_dimension, sizeof(Pixel));

    memcpy(imgResult->pixels, img->pixels, array_dimension * sizeof(Pixel));//copy the array img->pixels to imgResult->pixels

    return imgResult;
}

void ReadSecretKeyFromFile(char* fisier_cheie, unsigned int* r_seed, unsigned int* starting_value)
{
    FILE * f;
    f = fopen(fisier_cheie, "r");
    fscanf(f, "%u %u", r_seed, starting_value);
    fclose(f);
}

void ImageEncryption(char* file_initial_image, char* file_encrypted_image, char* file_secret_key)
{
    Image* img = UploadImageToMemory(file_initial_image);
    //printf("RGB0: %u,%u,%u\n", img->data[0].r, img->data[0].g, img->data[0].b);
    //printf("RGB1: %u,%u,%u\n", img->data[1].r, img->data[1].g, img->data[1].b);
    //printf("RGB5: %u,%u,%u\n", img->data[5].r, img->data[5].g, img->data[5].b);
    unsigned int r_seed;
    unsigned int starting_value;//SV
    ReadSecretKeyFromFile(file_secret_key, &r_seed, &starting_value);


    unsigned int seed = r_seed;
    int i=0;
    int array_dimension_random_numbers = 2 * img->header.width_px * img->header.height_px - 1;
    unsigned int* R = (unsigned int*) calloc(array_dimension_random_numbers, sizeof(unsigned int));

    for(i=0; i<array_dimension_random_numbers; i++)
    {
        unsigned int RandomNumber = XorShift32(&seed);
        R[i]=RandomNumber;
    }

     ///Create sigma array values with: 0..n
    unsigned int array_dimension_sigma = img->header.width_px * img->header.height_px;
    unsigned int* Sigma = (unsigned int*) calloc(array_dimension_sigma, sizeof(unsigned int));
    for(i=0; i<array_dimension_sigma; i++)
    {
        Sigma[i] = i;
    }

    ///Permutation array sigma values
    unsigned int j, a=0;
    unsigned int tmp;
    unsigned int len = array_dimension_sigma;

    while(len)
    {
        j = R[a]%(len);

        if (j != len - 1)
        {
            tmp = Sigma[j];
            Sigma[j] = Sigma[len - 1];
            Sigma[len - 1] = tmp;
        }
        a++;
        len--;
    }

    ///Clone the initial image and set the pixels value to 0
    Image* imgInt = CloneImage(img);
    memset(imgInt->pixels, 0, imgInt->header.width_px * imgInt->header.height_px * sizeof(Pixel));// put 0 in whole array imgFin->pixels
    unsigned int image_dimension = imgInt->header.width_px * imgInt->header.height_px;

    for(i=0; i<image_dimension; i++)
    {
        imgInt->pixels[Sigma[i]] = img->pixels[i];
    }

    WriteImageOnDisk(imgInt, "IntermediateImageEncryption.bmp");

    Image* imgFin = CloneImage(img);
    memset(imgFin->pixels, 0, imgFin->header.width_px * imgFin->header.height_px * sizeof(Pixel));// put 0 in whole array imgFin->pixels


    imgFin->pixels[0].b = ((unsigned int)starting_value) ^ imgInt->pixels[0].b ^ ((unsigned int)R[image_dimension]);
    imgFin->pixels[0].g = ((unsigned int)(starting_value >> 8)) ^ imgInt->pixels[0].g ^ ((unsigned int)(R[image_dimension] >> 8));
    imgFin->pixels[0].r = ((unsigned int)(starting_value >> 16)) ^ imgInt->pixels[0].r ^ ((unsigned int)(R[image_dimension] >> 16));

    for(i=1; i<image_dimension; i++)
    {
        imgFin->pixels[i].b = imgFin->pixels[i-1].b ^ imgInt->pixels[i].b ^ ((unsigned int)R[image_dimension+i]);
        imgFin->pixels[i].g = imgFin->pixels[i-1].g ^ imgInt->pixels[i].g ^ ((unsigned int)(R[image_dimension+i] >> 8));
        imgFin->pixels[i].r = imgFin->pixels[i-1].r ^ imgInt->pixels[i].r ^ ((unsigned int)(R[image_dimension+i] >> 16));
    }

    WriteImageOnDisk(imgFin, file_encrypted_image);

    free(imgInt);
    free(imgFin);
}





/// 5: DECRYPT AN ENCRYPTED BMP IMAGE

void ImageDecryption(char* file_encrypted_image, char* file_decrypted_image, char* file_secret_key)
{
    Image* img = UploadImageToMemory(file_encrypted_image);

    unsigned int r_seed;
    unsigned int starting_value;
    ReadSecretKeyFromFile(file_secret_key, &r_seed, &starting_value);

    unsigned int seed = r_seed;
    int i=0;
    int array_dimension_random_numbers = 2 * img->header.width_px * img->header.height_px - 1;
    unsigned int* R = (unsigned int*) calloc(array_dimension_random_numbers, sizeof(unsigned int));

    for(i=0; i<array_dimension_random_numbers; i++)
    {
        unsigned int RandomNumber = XorShift32(&seed);
        R[i]=RandomNumber;
    }

     ///Create the sigma array with values: 0..n
    unsigned int array_dimension_sigma = img->header.width_px * img->header.height_px;
    unsigned int* Sigma = (unsigned int*) calloc(array_dimension_sigma, sizeof(unsigned int));
    for(i=0; i<array_dimension_sigma; i++)
    {
        Sigma[i] = i;
    }

    ///Permutation array values sigma

    unsigned int j,a=0;
    unsigned int tmp;
    unsigned int len = array_dimension_sigma;

    while(len)
    {
        j = R[a]%(len);

        if (j != len - 1)
        {
            tmp = Sigma[j];
            Sigma[j] = Sigma[len - 1];
            Sigma[len - 1] = tmp;
        }
        a++;
        len--;
    }

    ///We clone the initial image and we set the pixels value to 0
    Image* imgInt = CloneImage(img);
    memset(imgInt->pixels, 0, imgInt->header.width_px * imgInt->header.height_px * sizeof(Pixel));
    unsigned int image_dimension = imgInt->header.width_px * imgInt->header.height_px;

    imgInt->pixels[0].b = ((unsigned int)starting_value) ^ img->pixels[0].b ^ ((unsigned int)R[image_dimension]);
    imgInt->pixels[0].g = ((unsigned int)(starting_value >> 8)) ^ img->pixels[0].g ^ ((unsigned int)(R[image_dimension] >> 8));
    imgInt->pixels[0].r = ((unsigned int)(starting_value >> 16)) ^ img->pixels[0].r ^ ((unsigned int)(R[image_dimension] >> 16));

    for(i=1; i<image_dimension; i++)
    {
        imgInt->pixels[i].b = img->pixels[i-1].b ^ img->pixels[i].b ^ ((unsigned int)R[image_dimension+i]);
        imgInt->pixels[i].g = img->pixels[i-1].g ^ img->pixels[i].g ^ ((unsigned int)(R[image_dimension+i] >> 8));
        imgInt->pixels[i].r = img->pixels[i-1].r ^ img->pixels[i].r ^ ((unsigned int)(R[image_dimension+i] >> 16));
    }

    WriteImageOnDisk(imgInt, "IntermediateImageDecryption.bmp");


    Image* imgFin = CloneImage(img);
    memset(imgFin->pixels, 0, imgFin->header.width_px * imgFin->header.height_px * sizeof(Pixel));

    for(i=0; i<image_dimension; i++)
    {
        imgFin->pixels[i] = imgInt->pixels[Sigma[i]];
    }

    WriteImageOnDisk(imgFin, file_decrypted_image);

    free(imgInt);
    free(imgFin);
}





/// 6: THE VALUES OF THE X^2 TEST FOR EACH COLOR CHANNEL OF A BMP IMAGE

double Formula(int *C, Image* img)
{
    int i = 255;
    double Frequency = (img->header.width_px * img->header.height_px) / 256.0;
    double sum = 0;
    while(i>=0)
    {
        sum = sum + pow(C[i]-Frequency, 2) / Frequency;
        i--;
    }
    return sum;
}

void ChiPatrat(char* source_file)
{
    Image* img = UploadImageToMemory(source_file);

    int *R, *G, *B; //arrays that count the frequencies of the color channel's 256 values
    R = (int*)calloc(256, sizeof(int));
    G = (int*)calloc(256, sizeof(int));
    B = (int*)calloc(256, sizeof(int));

    for( int i = 0 ; i < img->header.height_px ; i++)
    {
        for( int j = 0 ; j < img->header.width_px ; j++)
        {
            int index = i * img->header.width_px + j;
            R[img->pixels[index].r]++;
            G[img->pixels[index].g]++;
            B[img->pixels[index].b]++;
        }
    }
    printf("( %.2f, %.2f, %.2f )\n", Formula(R, img), Formula(G, img), Formula(B, img));
}





/// 7: TEMPLATE MATCHING BETWEEN: AN IMAGE AND A TEMPLATE

void grayscale_image(char* source_file_name,char* destination_file_name)
{
    FILE *fin, *fout;
    unsigned int dim_img, width_img, height_img;
    unsigned char pRGB[3], aux;

    printf("source_file_name = %s \n",source_file_name);

    fin = fopen(source_file_name, "rb");
    if(fin == NULL)
    {
        printf("We didn't find the source image from which we read");
        return;
    }

    fout = fopen(destination_file_name, "wb+");

    fseek(fin, 2, SEEK_SET);
    fread(&dim_img, sizeof(unsigned int), 1, fin);
    printf("The dimension of the image in bytes: %u\n", dim_img);

    fseek(fin, 18, SEEK_SET);
    fread(&width_img, sizeof(unsigned int), 1, fin);
    fread(&height_img, sizeof(unsigned int), 1, fin);
    printf("The dimension of the image in pixels (width x height): %u x %u\n",width_img, height_img);

    //we copy the initial image to the new one byte by byte
    fseek(fin,0,SEEK_SET);
    unsigned char c;
    while(fread(&c,1,1,fin)==1)
    {
        fwrite(&c,1,1,fout);
        fflush(fout);
    }
    fclose(fin);

    //we calculate the padding for a single line
    int padding;
    if(width_img % 4 != 0)
        padding = 4 - (3 * width_img) % 4;
    else
        padding = 0;

    printf("padding = %d \n",padding);

    fseek(fout, 54, SEEK_SET);
    int i,j;
    for(i = 0; i < height_img; i++)
    {
        for(j = 0; j < width_img; j++)
        {
            //we read the colors of the pixel
            fread(pRGB, 3, 1, fout);
            //we make the conversion in gray pixel
            aux = 0.299*pRGB[2] + 0.587*pRGB[1] + 0.114*pRGB[0];
            pRGB[0] = pRGB[1] = pRGB[2] = aux;
            fseek(fout, -3, SEEK_CUR);
            fwrite(pRGB, 3, 1, fout);
            fflush(fout);
        }
        fseek(fout,padding,SEEK_CUR);
    }
    fclose(fout);
}
unsigned char IntensityPx(FILE *f)
{
    Pixel x;
    fread(&x, sizeof(Pixel), 1, f);
    return x.r;
}
double Average(FILE *f)
{
    Pixel x;
    double Average;
    Average = 0;
    Template s;
    for(int i =0; i < s.height; i++)
    {
        for(int j = 0; j < s.width; j++)
        {
            fread(&x, sizeof(Pixel), 1, f);
            Average = Average + x.r;
        }
    }
    Average = Average / 165;
    return Average;
}
double Deviation(FILE *f)
{
    double Deviation;
    Deviation = 0;
    Template s;
    for(int i =0; i < s.height; i++)
    {
        for(int j = 0; j < s.width; j++)
        {
            Deviation = Deviation + (IntensityPx(f) - Average(f));
        }
    }
    Deviation = sqrt((1 / (s.area - 1)) * Deviation);
    return Deviation;
}
int Correlation(int n, FILE *I, FILE *S, int x, int width, int height)
{
    Image *img;
    Pixel p;
    Template s;
    if((img->header.width_px - width) < s.width)
    {
        exit(EXIT_FAILURE);
    }
    if((img->header.height_px - height) < s.height)
    {
        exit(EXIT_FAILURE);
    }
    fseek(I, 0, SEEK_SET);
    for(int i = 0; i < x; i++)
        fread(&p, sizeof(Pixel), 1, I);
    double WindowDeviation, TemplateDeviation, WindowAverage, AverageTemplate, sum;
    WindowDeviation = Deviation(I);
    TemplateDeviation = Deviation(S);
    WindowAverage = Average(I);
    AverageTemplate = Average(S);
    fseek(I, -s.area, SEEK_CUR);
    fseek(S, -s.area, SEEK_CUR);
    for(int i = 0; i < s.height; i++)
    {
        for(int j = 0; j < s.width; j++)
        {
            sum = (1 / (WindowDeviation * TemplateDeviation)) * (IntensityPx(I) - WindowAverage) * (IntensityPx(S) - AverageTemplate);
        }
    }
    fclose(I);
    fclose(S);
    return (1 / n * sum);
}

void TemplateMatching(char* image, char* template1)
{
    /// Grayscale: Image and Template
    char* image_grayscale = "ImgGrayscale.bmp";
    grayscale_image(image, image_grayscale);
    char* Template_grayscale = "PatternGrayscale.bmp";
    grayscale_image(template1, Template_grayscale);

    /// Detections
    Image* img = UploadImageToMemory(image_grayscale);
    Template s;
    s.width = 11;
    s.height = 15;
    s.area = s.height * s.width;
    double verge;
    verge = 0.5;
    int x;//corner - top left of the window(the number of the pixel)
    int y;//corner - bottom right of the window(the number of the pixel)
    FILE *I, *S;
    I = fopen(image_grayscale, "rb");
    if(I == NULL)
    {
        printf("We didn't find the image!");
        return ;
    }
    S = fopen(Template_grayscale, "rb");
    if(S == NULL)
    {
        printf("We didn't find the image!");
        return ;
    }
    for(int i = 0; i < img->header.height_px; i += s.height)
    {
        for(int j = 0; j < img->header.width_px; j += s.width)
        {
            if(Correlation(s.area, I, S, (i / s.height) * img->header.width_px + j, j, i) > verge) // if it's detection
            {
                //retain the corners of the detection
                x = (i / s.height) * img->header.width_px + j;
                y = x + img->header.width_px * i + s.width;
            }

        }
    }
    fclose(I);
    fclose(S);
}

int main()
{
    char *file_secret_key = "secret_key.txt";
    char *file_original_image = "peppers.bmp";
    char *file_encrypted_image = "peppers_encrypted.bmp";
    char *file_decrypted_image = "peppers_decrypted.bmp";

    ImageEncryption(file_original_image, file_encrypted_image, file_secret_key);
    ImageDecryption(file_encrypted_image, file_decrypted_image, file_secret_key);

    printf("X^2 Original Image\n");
    ChiPatrat(file_original_image);

    printf("X^2 Encrypted Image\n");
    ChiPatrat(file_encrypted_image);

    char *image = "test.bmp";
    char *template1 = "figure0.bmp";
    TemplateMatching(image, template1);
    system("pause");

    return 0;
}
