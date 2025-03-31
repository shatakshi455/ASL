#include <stdio.h>

int main(){
    char example[7] = {'m', 't', 'm', 'n', 'o', 'p', '\n'};
    char *p = example;
    printf("%c", *p);
    *(p + 3) = 'k';
    *(++p) = 'm';
    printf("%s", example);
}