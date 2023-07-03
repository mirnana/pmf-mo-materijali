**Zadatak**. Napisati program koji računa axpy operaciju na grafičkoj kartici.
Axpy (a x plus y) je operacija "z = a*x +y", gdje su "x", "y" i "z" vektori, a "a" je skalari(#).
Potrebno je:

- Alocirati i inicijalizirati x i y na CPU te ih prebaciti na GPU (zadati si skalar a).
- Izvršiti axpy jezgru na GPU i vratiti polje "z" na CPU. 
- Izmjeriti vrijeme koje je uzelo izvršavanje jezgre. 
- Napisati axpy funkciju koja će istu operaciju izvršiti na CPU i izmjeriti potrebno vrijeme.
- Provjeriti da GPU i CPU daju isti rezultat (eventualno do na greške zaokruživanja). 
- Ispisati vremena izvršavanja na GPU i na CPU. 

Izmjereno vrijeme mora biti srednje vrijeme izračunato na 16 izvršavanja.
Sva polja neka budu dimenzije 80.000.000 elemenata. Paziti da ispravno dealocirate memoriju. 

Kako se ponaša vrijeme izvršavanja na GPU s dimenzijom bloka?

---
(#) Pravi axpy je "y=a*x+y", ali ovdje radi jednostavnosti provjere rezultata koristimo dodatno 
polje "z".
