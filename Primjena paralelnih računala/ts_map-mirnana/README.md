 **Zadatak**. Treba napraviti asocijativno polje sigurno za istodobni pritup (_thread safe_). 
 Spremnici poput `std::map<>`, `std::unordered_map<>` i drugi
  nisu sigurni za konkurentni pristup. Najveći problem pri istodobnom pristupu predstavljaju 
  iteratori. Dovoljno je zamisliti situaciji u kojoj dvije niti drže iteratore na isti element pri čemu
  jedna nit mijenja element, a druga ga pokušava izbrisati. 
  Kako bismo pojednostavili zadaću konstrukcije asocijativnog polja izbacit ćemo iteratore iz sučelja. 
 
  U sučelju ćemo imati samo ove 4 operacije:
  1. insert(): Dodaj par ključ-vrijednost u spremnik.
  2. assign(): Promijeni vrijednost za dani ključ.
  3. get():    Dohvati vrijednost pridruženu ključu ako je ključ u spremniku.
  4. remove(): Obriši ključ-vrijednost par iz spremnika. 
 
 **Implementacija**.

  Tri su najčešća načina implementacije asocijativnog polja: binarno stablo, sortirano polje i _hash_ tablica. 
  Kod binarnog stabla uvijek moramo zaključati korijenski čvor. Kod prijelaza na drugi čvor možemo ga 
  otključati i zaključati mutex pridružen drugom čvoru, no mogućnosti za fino-granulirani 
  paralelizam su male. Slična je situacija i sa sortiranim poljem. Stoga nam ostaje _hash_ tablica.
 
  Implementirat ćemo _hash_ tablicu **s fiksnim brojem pretinaca**. Svaki pretinac sadrži listu ključ-vrijednost
  parova. Za listu ćemo koristiti `std::list` spremik. Svaki pretinac će biti štićen s jednim lokotom tipa 
  `std::shared_mutex<>` kako bi se iskoristila činjenica da se asocijativno polje često više čita nego mijenja. 
 
 **Paralelna implementacija**.

  Svaki pretinac je zaštićen svojim lokotom,tako da se različitim pretincima može pristupati paralelno. 
  `std::hash` funkcijski objekt ima operator funkcijskog poziva koji je `const` i stoga STL garantira da je 
  siguran za istovremene pozive. Stoga haširanje ne trebamo štiti lokotima. 
 
  Pretince štitimo sa `std::shared_mutex` tipom jer očekujemo da će ova struktura biti puno češće 
  čitana nego modificirana. Sučelje ima ove i samo ove metode:
 
  - `insert()` metoda ubacuje par u spremik ako njegov ključ još nije prisutan u spremniku. Ako kljuć već 
    jeste u spremniku metoda  ne radi ništa i vraća `false`.
    Kada ubaci element vraća `true`. 
  - `assign()` metoda mijenja vrijednost za postojeći ključ. Ako ključ nije prisutan ne radi ništa i vraća `false`.
    Kada promijeni vrijednost vraća `true`. 
  - `remove()` eliminira element danog ključa i vraća `true`.  Ako ključ nije prisutan ne radi ništa i vraća `false`.
  - `get()` vraća vrijednost za dani ključ. Povratna vrijednost je `std::optional` koji je prazan ako ključ 
    nije prisutan. 
  - `print()` metoda ispisuje čitav spremnik, uključujući pune i prazne pretince. 
  - `size()` metoda vraća broj prisutnih elemenata u spremniku. 
 
Pri ispisu spremnika treba zaključati sve pretince kako se spremnik ne bi mijenjao za vrijeme ispisa. 
 
Za ispis broja elemenata spremnika (metoda `size()`) koristiti varijablu koja prati trenutani broj 
elemenata. Zaštititi varijablu lokotom. 

Struktira se zove `Map<>` i kako je parametrizirana cijela je implementirana u datoteci `ts_map.h`.
Sve metode klase `Map<>` moraju biti implementirani **izvan** klase. 

Napisati main program u `ts_map_main.cpp` datoteci. Program mora što bolje testirati čitavu strukturu paralelnom
načinu. 

Ne uvoditi nove datoteke niti ne mijenjati imena datoteka. 
