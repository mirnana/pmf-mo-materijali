#ifndef __PROTOCOL_H_
#define __PROTOCOL_H_

#define POTEZ       	1
#define GRESKA      	2
#define KRAJ        	3
#define STANJE          4
#define IGRAJ           5
#define IDUCI           6
#define DOBRO           7

#define OK      1
#define NIJEOK  0

int primiPoruku(int sock, int* vrstaPoruke, int* brojIgre, char** poruka);
int posaljiPoruku(int sock, int vrstaPoruke, int brojIgre, const char* poruka);
/*
#define error1(s){
    printf(s);
    exit(0);
}

#define error2(s1, s2){ 
    printf(s1, s2); 
    exit(0); 
}

#define myperror(s){ 
    perror(s); 
    exit(0); 
}
*/
#endif
