#include <stdio.h>
#include <stdlib.h>

int main() {
    int x[8][6] = {{0,0,0,1,0,1}, 
                   {1,0,0,0,0,0},
                   {1,0,1,0,1,0},
                   {1,1,0,0,1,1},
                   {1,1,1,0,0,0},
                   {1,1,1,1,0,0},
                   {1,1,1,1,1,0},
                   {1,1,1,1,1,1}};

    int vp = 0, vn = 0, fp = 0, fn = 0;

    // Primero definimos la salida esperada
    int Yd[8] = {1,0,1,0,1,1,0,0};

    // Definir Y(obt) solamente para {este caso}
    int Yobt[8] = {1,0,0,1,1,1,0,0};

    // Obtenemos el número de filas y columnas
    int num_filas = sizeof(x) / sizeof(x[0]);
    int num_columnas = sizeof(x[0]) / sizeof(x[0][0]);

    int tamanio_Yd = sizeof(Yd)/sizeof(Yd[0]);
    int tamanio_Yobt = sizeof(Yobt)/sizeof(Yobt[0]);

    if(tamanio_Yd == tamanio_Yobt){
        for(int i=0; i<tamanio_Yd; i++){
            if(Yd[i] == 1 && Yobt[i] == 1)
                vp+=1;
            if(Yd[i] == 0 && Yobt[i] == 0)
                vn+=1;
            if(Yd[i] == 1 && Yobt[i] == 0)
                fn+=1;
            if(Yd[i] == 0 && Yobt[i] == 1)
                fp+=1;
        }
    }

  /*   printf("%d\n", vp);
    printf("%d\n", vn);
    printf("%d\n", fn);
    printf("%d\n", fp); */

    float P = (float)vp / (float)(vp + fp);
    float Ex = (float)(vp + vn) / (float)(vp + vn + fp +fn);
    float R = (float)vp / (float)(vp + fn);
    float F2 = 2 * P * R / (P + R);


    printf("Presicion: %.2f\n", P);
    printf("Exactitud: %.2f\n", Ex);
    printf("Recall: %.2f\n", R);
    printf("F2 Score: %.2f\n", F2);

    return 0;
}
