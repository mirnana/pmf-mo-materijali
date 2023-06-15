#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>

#include <pthread.h>

#include "protocol.h"

#define MAXDRETVI 15

typedef struct{
	int commSocket;
	int indexDretve;
    int brojIgre;
	int prvi;
} obradiKlijenta__parametar;

int aktivneDretve[MAXDRETVI] = {0};
obradiKlijenta__parametar parametarDretve[MAXDRETVI];
pthread_mutex_t lokot_aktivneDretve = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t lokot_tablice = PTHREAD_MUTEX_INITIALIZER;

void krajKomunikacije(void* parametar);
void* obradiKlijenta(void* parametar);

typedef struct{
	int a;
	int b;
	int c;
} int3;

typedef struct{
	char* prva;
	char* druga;
} char2;

//static int** sibice;
static int3* sibice;
static int* dostupan; 
static char2* adrese;
static int* pobjednik;
static int* sljedeci;
static int A, B, C;

int main(int argc, char **argv){
	if(argc != 6){
		printf("Upotreba: %s port n A B C\n", argv[0]);
		exit(0);
	} 
	
	int port, n;
	sscanf(argv[1], "%d", &port);
    sscanf(argv[2], "%d", &n);
    if(n > 7){
		printf("Odabrani n je prevelik! Imam dovoljno dretvi za najviše 7 parova.\n");
		exit(0);
	} 
    sscanf(argv[3], "%d", &A);
    sscanf(argv[4], "%d", &B);
    sscanf(argv[5], "%d", &C);
	
	sibice = (int3*) malloc(n * sizeof(int3));
	dostupan = (int*) malloc(n * sizeof(int));
	adrese = (char2*) malloc(n * sizeof(char2));
	pobjednik = (int*) malloc(n * sizeof(int));
	sljedeci = (int*) malloc(n * sizeof(int));
	int i;
    for(i = 0; i < n; i++){
		sibice[i].a = A;
		sibice[i].b = B;
		sibice[i].c = C;
        dostupan[i]  = 1; 
        adrese[i].prva = NULL;
        adrese[i].druga = NULL;
        pobjednik[i] = 0;
		sljedeci[i] = 0;
    }
	int listenerSocket = socket(PF_INET, SOCK_STREAM, 0);
	if(listenerSocket == -1){
		perror("socket");
		exit(0);
	}

	struct sockaddr_in mojaAdresa;
	mojaAdresa.sin_family      = AF_INET;
	mojaAdresa.sin_port        = htons(port);
	mojaAdresa.sin_addr.s_addr = INADDR_ANY;
	memset( mojaAdresa.sin_zero, '\0', 8 );

	if(bind(listenerSocket,	(struct sockaddr *) &mojaAdresa, sizeof(mojaAdresa)) == -1){
		perror("bind");
		exit(0);
	}

	if(listen(listenerSocket, 10) == -1){
		perror("listen");
		exit(0);
	}

	pthread_t dretve[15];

	while(1){
		struct sockaddr_in klijentAdresa;
		unsigned int lenAddr = sizeof(klijentAdresa);
		int commSocket = accept(listenerSocket, (struct sockaddr *) &klijentAdresa, &lenAddr);
		if(commSocket == -1){
			perror("accept");
			exit(0);
		}

		char *dekadskiIP = inet_ntoa(klijentAdresa.sin_addr);
		printf("Prihvatio konekciju od %s ", dekadskiIP);        

		pthread_mutex_lock(&lokot_aktivneDretve);
		int i, indexNeaktivne = -1;
		for(i = 0; i < MAXDRETVI; ++i){
			if(aktivneDretve[i] == 0) indexNeaktivne = i;
			else if(aktivneDretve[i] == 2){
				pthread_join(dretve[i], NULL);
				aktivneDretve[i] = 0;
				indexNeaktivne = i;
			}
        }
		if(indexNeaktivne == -1){
			close( commSocket ); 
			printf( "--> ipak odbijam konekciju jer nemam vise dretvi.\n" );
		}
		else{
            pthread_mutex_lock(&lokot_tablice);
            //potraži slobodan brojIgre u int dostupan[n]; 
            int brojIgre;
			int i;
            for(i = 0; i < n; i++)
                if(dostupan[i] == 1){
                    brojIgre = i;
                    break;
                }			
            //potraži slobodno mjesto u char adrese[brojIgre][]; i smjesti tamo dekadskiIP
            //ako je sada brojIgre zauzet sa 2 igrača, isto evidentiramo u dostupan[brojIgre] te šaljemo klijentima poruku STANJE
            if(adrese[brojIgre].prva == NULL){				
				adrese[brojIgre].prva = dekadskiIP;
				aktivneDretve[indexNeaktivne] = 1;
				parametarDretve[indexNeaktivne].commSocket = commSocket;
				parametarDretve[indexNeaktivne].indexDretve = indexNeaktivne;
				parametarDretve[indexNeaktivne].brojIgre = brojIgre;
				parametarDretve[indexNeaktivne].prvi = 1; //istina je da je prvi na redu
				printf( "--> koristim dretvu broj %d.\n", indexNeaktivne );
			} 
            else{				
                adrese[brojIgre].druga = dekadskiIP;
                dostupan[brojIgre] = 0;
				aktivneDretve[indexNeaktivne] = 1;
				parametarDretve[indexNeaktivne].commSocket = commSocket;
				parametarDretve[indexNeaktivne].indexDretve = indexNeaktivne;
				parametarDretve[indexNeaktivne].brojIgre = brojIgre;
				parametarDretve[indexNeaktivne].prvi = 0; //nije prvi na redu
				printf( "--> koristim dretvu broj %d.\n", indexNeaktivne );
            }
            
            pthread_mutex_unlock(&lokot_tablice);
			pthread_create(&dretve[indexNeaktivne], NULL, obradiKlijenta, &parametarDretve[indexNeaktivne]);			
		}
		pthread_mutex_unlock(&lokot_aktivneDretve);
	}
	return 0;
}


void* obradiKlijenta(void* parametar){
	obradiKlijenta__parametar *param = (obradiKlijenta__parametar *) parametar;
	int commSocket = param -> commSocket;
	int brojIgre = param -> brojIgre;
	int prvi = param -> prvi;
	int tmp1, tmp2;
	int gotovo = 0;
	while(!gotovo){
        char poruka[10] = {};
		char* poruka2;		
        sprintf(poruka, "%d %d %d", sibice[brojIgre].a, sibice[brojIgre].b, sibice[brojIgre].c);
        if(posaljiPoruku(commSocket, STANJE, brojIgre, poruka) == NIJEOK){
			printf("Error STANJE");
        	krajKomunikacije(param);
            return NULL;
		}
		if(prvi){		
			while(dostupan[brojIgre] != 0){} //strpi se
			if(pobjednik[brojIgre] != 0) {printf("bruh\n"); goto OVDJE1;}
			//šalji IGRAJ
			if(posaljiPoruku(commSocket, IGRAJ, brojIgre, "") == NIJEOK){
				printf("Error IGRAJ");
                krajKomunikacije(param);
                return NULL;
            }
			//očekuj POTEZ
			if(primiPoruku(commSocket, &tmp1, &tmp2, &poruka2) == NIJEOK){
				printf("Error primiPoruku");
				krajKomunikacije(param);
				return NULL;
			}
			if(tmp1 != POTEZ || tmp2 != brojIgre){
				krajKomunikacije(param);
				return NULL;
			}
			int hrpa, broj;
			sscanf(poruka2, "%d %d", &hrpa, &broj);
			//ako potez nije validan, šalji GRESKA
			while((hrpa != 1 && hrpa != 2 && hrpa != 3) || (hrpa == 1 && sibice[brojIgre].a < broj)
					|| (hrpa == 2 && sibice[brojIgre].b < broj) || (hrpa == 3 && sibice[brojIgre].c < broj)){ //ako potez nije validan, posalji
				if(posaljiPoruku(commSocket, GRESKA, brojIgre, "") == NIJEOK){				   //poruku o gresci i trazi novi potez
                	printf("Error GRESKA");
					krajKomunikacije(param);
                	return NULL;
            	}
				//ocekuj novi POTEZ
				if(primiPoruku(commSocket, &tmp1, &tmp2, &poruka2) == NIJEOK){
					printf("Error primiPoruku");
					krajKomunikacije(param);
					return NULL;
				}
				if(tmp1 != POTEZ || tmp2 != brojIgre){
					krajKomunikacije(param);
					return NULL;
				}
				sscanf(poruka2, "%d %d", &hrpa, &broj);
			}
			//ako je potez validan, salji DOBRO i evidentiraj
			if(posaljiPoruku(commSocket, DOBRO, brojIgre, "") == NIJEOK){
				printf("Error DOBRO");
                krajKomunikacije(param);
                return NULL;
            }
			sljedeci[brojIgre]++;
			pthread_mutex_lock(&lokot_tablice);
			switch(hrpa){
				case 1:{
					sibice[brojIgre].a -= broj;
					break;
				}
				case 2:{
					sibice[brojIgre].b -= broj;
					break;
				}
				default:{
					sibice[brojIgre].c -= broj;
					break;
				}
			}
			pthread_mutex_unlock(&lokot_tablice);
			//očekuj poruku IDUCI
			while(sljedeci[brojIgre] % 2 == 1){}
			//provjeri stanje i pošalji ili STANJE ili KRAJ
			OVDJE1:	//label
			pthread_mutex_lock(&lokot_tablice);
			int flag = 1;
			if(sibice[brojIgre].a == 0 && sibice[brojIgre].b == 0 && sibice[brojIgre].c == 0){
				if(pobjednik[brojIgre] == 0){ //stanje u pobjednik[] je nepromijenjeno, dakle ovaj igrač je gubitnik
					pobjednik[brojIgre]++;
					sljedeci[brojIgre]++;
					if(posaljiPoruku(commSocket, KRAJ, brojIgre, "0") == NIJEOK){
						printf("Error KRAJ");
            	   		krajKomunikacije(param);
                		return NULL;
	            	}
					flag = 0;
					pthread_mutex_unlock(&lokot_tablice);
					krajKomunikacije(param);
					return NULL;
				}
				else if(flag){						//ovaj igrač je pobijedio
					if(posaljiPoruku(commSocket, KRAJ, brojIgre, "1") == NIJEOK){
						printf("Error KRAJ");
            	   		krajKomunikacije(param);
                		return NULL;
	            	}	
					sibice[brojIgre].a = A;
		        	sibice[brojIgre].b = B;
	    		    sibice[brojIgre].c = C;
    	    		dostupan[brojIgre]  = 1;
	    	    	adrese[brojIgre].prva = NULL;
	    	    	adrese[brojIgre].druga= NULL;
    	    		pobjednik[brojIgre] = 0;
					sljedeci[brojIgre] = 0;
					pthread_mutex_unlock(&lokot_tablice);
					krajKomunikacije(param);
					return NULL;
				}
			}
			else{
				sprintf(poruka, "%d %d %d", sibice[brojIgre].a, sibice[brojIgre].b, sibice[brojIgre].c);
				if(posaljiPoruku(commSocket, STANJE, brojIgre, poruka) == NIJEOK){
					printf("Error STANJE");
                	krajKomunikacije(param);
                	return NULL;
            	}
			}
			pthread_mutex_unlock(&lokot_tablice);
		}
		else{
			//očekuj poruku IDUCI	
			while(sljedeci[brojIgre] % 2 == 0){} //cekamo prije poruke IDUCI potvrdu da je red na nas valjdddd	
			//provjeri stanje i pošalji ili STANJE ili KRAJ
			pthread_mutex_lock(&lokot_tablice);		
			int flag = 1;
			if(sibice[brojIgre].a == 0 && sibice[brojIgre].b == 0 && sibice[brojIgre].c == 0){
				printf("usao u veliki if kod KRAJ\n");
				if(pobjednik[brojIgre] == 0){ //stanje u pobjednik[] je nepromijenjeno, dakle ovaj igrač je gubitnik
					pobjednik[brojIgre]++;
					sljedeci[brojIgre]++;
					flag = 0;
					pthread_mutex_unlock(&lokot_tablice);
					if(posaljiPoruku(commSocket, KRAJ, brojIgre, "0") == NIJEOK){
						printf("Error KRAJ");
            	   		krajKomunikacije(param);						   
                		return NULL;
	            	}										
					krajKomunikacije(param);
					return NULL;
				}
				else if(flag){						//ovaj igrač je pobijedio
					if(posaljiPoruku(commSocket, KRAJ, brojIgre, "1") == NIJEOK){
						printf("Error KRAJ");
            	   		krajKomunikacije(param);
                		return NULL;
	            	}						
					sibice[brojIgre].a = A;
		        	sibice[brojIgre].b = B;
	    		    sibice[brojIgre].c = C;
    	    		dostupan[brojIgre] = 1;
	    	    	adrese[brojIgre].prva = NULL;
	    	    	adrese[brojIgre].druga = NULL;
    	    		pobjednik[brojIgre] = 0;
					pthread_mutex_unlock(&lokot_tablice);
					krajKomunikacije(param);
					return NULL;
				}
			}			
			else{
				sprintf(poruka, "%d %d %d", sibice[brojIgre].a, sibice[brojIgre].b, sibice[brojIgre].c);
				if(posaljiPoruku(commSocket, STANJE, brojIgre, poruka) == NIJEOK){
					printf("Error STANJE");
                	krajKomunikacije(param);
                	return NULL;
            	}
			}
			pthread_mutex_unlock(&lokot_tablice);
			//šalji IGRAJ			
			if(posaljiPoruku(commSocket, IGRAJ, brojIgre, "") == NIJEOK){
				printf("Error IGRAJ");
                krajKomunikacije(param);
                return NULL;
            }
			//očekuj POTEZ
			if(primiPoruku(commSocket, &tmp1, &tmp2, &poruka2) == NIJEOK){
				printf("Error primiPoruku");
				krajKomunikacije(param);
				return NULL;
			}
			if(tmp1 != POTEZ || tmp2 != brojIgre){
				printf("Error ");
				krajKomunikacije(param);
				return NULL;
			}
			int hrpa, broj;
			sscanf(poruka2, "%d %d", &hrpa, &broj);
			//ako potez nije validan, šalji GRESKA
			while((hrpa != 1 && hrpa != 2 && hrpa != 3) || (hrpa == 1 && sibice[brojIgre].a < broj)
					|| (hrpa == 2 && sibice[brojIgre].b < broj) || (hrpa == 3 && sibice[brojIgre].c < broj)){ //ako potez nije validan, posalji
				if(posaljiPoruku(commSocket, GRESKA, brojIgre, "") == NIJEOK){				   //poruku o gresci i trazi novi potez
                	printf("Error GRESKA");
					krajKomunikacije(param);
                	return NULL;
            	}
				if(posaljiPoruku(commSocket, IGRAJ, brojIgre, "") == NIJEOK){
					printf("Error IGRAJ");
	                krajKomunikacije(param);
    	            return NULL;
        	    }

				if(primiPoruku(commSocket, &tmp1, &tmp2, &poruka2) == NIJEOK){
					printf("Error primiPoruku");
					krajKomunikacije(param);
					return NULL;
				}
				if(tmp1 != POTEZ || tmp2 != brojIgre){
					printf("Error ");
					krajKomunikacije(param);
					return NULL;
				}
				sscanf(poruka2, "%d %d", &hrpa, &broj);
			}
			//ako je potez validan, salji DOBRO i evidentiraj
			if(posaljiPoruku(commSocket, DOBRO, brojIgre, "") == NIJEOK){
				printf("Error DOBRO");
                krajKomunikacije(param);
                return NULL;
            }
			sljedeci[brojIgre]++;
			pthread_mutex_lock(&lokot_tablice);
			switch(hrpa){
				case 1:{
					sibice[brojIgre].a -= broj;
					break;
				}
				case 2:{
					sibice[brojIgre].b -= broj;
					break;
				}
				default:{
					sibice[brojIgre].c -= broj;
					break;
				}
			}			
			pthread_mutex_unlock(&lokot_tablice);
		}
		free(poruka2);
	}
	return NULL;
}

void krajKomunikacije(void* parametar){
	obradiKlijenta__parametar *param = (obradiKlijenta__parametar *) parametar;
	int commSocket  = param -> commSocket;
	int indexDretve = param -> indexDretve;
    int brojIgre = param -> brojIgre;
	printf("Kraj komunikacije [dretva=%d, broj igre=%d]... \n", indexDretve, brojIgre);

	pthread_mutex_lock(&lokot_aktivneDretve);
	aktivneDretve[indexDretve] = 2;
	pthread_mutex_unlock(&lokot_aktivneDretve);

	close(commSocket);
}
