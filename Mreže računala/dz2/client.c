#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#include "protocol.h"

static int a, b, c;

int main(int argc, char **argv){
	if(argc != 3){
		printf("Upotreba: ./%s IP port\n", argv[0]);
		exit(0);
	}

	char *dekadskiIP = argv[1];

	int port;
	sscanf(argv[2], "%d", &port);

	int mojSocket = socket(PF_INET, SOCK_STREAM, 0);
	if(mojSocket == -1){
		perror("socket");
		exit(0);
	} 

	struct sockaddr_in adresaServera;
	adresaServera.sin_family = AF_INET;
	adresaServera.sin_port = htons(port);
	if(inet_aton(dekadskiIP, &adresaServera.sin_addr) == 0){
		printf("%s nije dobra adresa!\n", dekadskiIP);
		exit(0);
	} 
	memset(adresaServera.sin_zero, '\0', 8);

	if(connect(mojSocket, (struct sockaddr *) &adresaServera, sizeof(adresaServera)) == -1){
		perror("connect");
		exit(0);
	}

	int gotovo = 0;
	while(!gotovo){
		char poruka[10] = {};
		char* poruka2;
		int tmp1, tmp2;
		if(primiPoruku(mojSocket, &tmp1, &tmp2, &poruka2) == NIJEOK){
			printf("Error primiPoruku\n");
			exit(0);
		}
		switch(tmp1){
			case STANJE:{								
				sscanf(poruka2, "%d %d %d", &a, &b, &c);
				printf("Trenutno stanje šibica:\n1. ");
				int i;
				for(i = 0; i < a; i++) printf("i ");
				printf("\n2. ");
				for(i = 0; i < b; i++) printf("i ");
				printf("\n3. ");
				for(i = 0; i < c; i++) printf("i ");
				printf("\n");
				break;
			}
			case KRAJ:{
				int bla;
				sscanf(poruka2, "%d", &bla);
				if(bla == 1) printf("Čestitam, pobijedili ste!\n");
				else if(bla == 0) printf("Nažalost ste izgubili.\n");
				else printf("Error KRAJ");
				gotovo = 1;
				break;
			}
			case GRESKA:
			case IGRAJ:{
				printf("\n\nOdaberi redak (1, 2 ili 3):\n");
				int hrpa;
				scanf("%d", &hrpa);
				while((hrpa != 1 && hrpa != 2 && hrpa != 3) || (hrpa == 1 && a == 0) ||
						(hrpa == 2 && b == 0) || (hrpa == 3 && c == 0)){	
					printf("Pogrešan unos! Unesi 1, 2 ili 3:\n");
					scanf("%d", &hrpa);
				}
				int broj;
				printf("Unesi broj šibica koji želiš uzeti s navedene hrpe:\n");
				scanf("%d", &broj);
				while((hrpa == 1 && broj > a) || (hrpa == 2 && broj > b) || (hrpa == 3 && broj > c) || broj <= 0){ 
					printf("Potez nije validan! Probaj opet:\n");
					scanf("%d", &broj);
				}			
				sprintf(poruka, "%d %d", hrpa, broj);
				if(posaljiPoruku(mojSocket, POTEZ, tmp2, poruka) == NIJEOK){
					printf("Error POTEZ");
					exit(0);
				}
				break;
			}
			case DOBRO:{
				printf("Potez je evidentiran.\n");
				break;
			}
			default:{
				printf("Greška u pristignutoj poruci.\n");
				exit(0);
			}
		}
	}
	return 0;
}
