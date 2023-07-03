#  Zadatak

Želimo simulirati gibanje planeta (ili nekih drugih čestica) pod utjecajem gravitacijske 
sile. Stoga imamo $n$ tijela 
koje ćemo indeksirati indeksima od 0 do $n$ (isključivo). 
Pozicija i-tog tijela u trenutku $t$ neka je označena s $\mathbf{x}_i(t)$. 
Sila kojom tijelo $j$ djeluje na tijelo i dana je izrazom

$$ 
{\bf F}_{i,j} = - \frac{G m_i m_j}{| {\bf x}_i(t) - {\bf x}_j(t)|^3}  ({\bf x}_i(t) - {\bf x}_j(t)), 
\qquad (A) 
$$ 
 
gdje je 

- G gravitacijska konstanta $G = 6.673 \times 10^{−11}  m^3/(kg \cdot s^2)$.
- $m_i$ je masa $i$-tog tijela.

Ukupna sila na tijelo $i$ jednaka je:

$$ 
{\bf F}_i = \sum_{j \ne i} {\bf F}_{i,j} 
   = - G m_i \sum_{j \ne i} \frac{m_j}{|{\bf x}_i(t) - {\bf x}_j(t)|^3}  ({\bf x}_i(t) - {\bf x}_j(t)), 
$$ 

Time za gibanje sustava masa dobivamo sljedeći sustav diferencijalnih jednadžbi:

$$ 
\ddot{\mathbf{x}}_i(t) =  - G \sum_{j\ne i}
    \frac{m_j}{| {\bf x}_i(t) - {\bf x}_j(t)|^3}  ({\bf x}_i(t) - {\bf x}_j(t)), \quad i=0,1,\ldots, n-1.
$$

Diferencijalnim jednadžbama trebamo dodati i *početne uvjete*:

$$  
{\bf x}_i(0) = {\bf x}_i^0,\quad \dot{\bf x}_i(0) = {\bf v}_i^0, \quad i=0,1,\ldots n-1.
$$
   

Iskoristit ćemo eksplicitnu Eulerovu metodu: uzimamo vremensku mrežu s konstantnim vremenskim 
korakom $\delta t$ i definiramo $t^n = n \delta t$, te $\mathbf{x}_i^n = \mathbf{x}_i(t^n)$. 
Pratimo poziciju i brzinu svih čestica. Uz oznaku ${\bf v}_i^n = \dot{\bf x}_i(t^n)$ imamo
sljedeću aproksimaciju:

$$  
\mathbf{x}_i^{n+1} = \mathbf{x}_i^n + \delta t \mathbf{v}_i^n,\quad
    \mathbf{v}_i^{n+1} = \mathbf{v}_i^n + \delta t {\bf F}_i/m_i.
$$


*Kolizije*. Problem ćemo pojednostaviti tako što nećemo uzeti u obzir eventualne kolizije između 
tijela. Kako ne bismo dobili  beskonačne sile u trenucima kolizije (odnosno NaN-ove) uvest ćemo 
parametar $l >0$ i zamijenit ćemo $|\mathbf{x}_i(t) - \mathbf{x}_j(t)|$ u (A)  izrazom 
$| \mathbf{x}_i(t) - \mathbf{x}_j(t) | + l$. 

## Struktura programa 

- Potrebno je napraviti samo *sekvencijalni* program (u prvom koraku). 
- Funkcija `main()` je zadana. U jednoj petlji računamo sile, zatim pomičemo sve čestice i
nakon toga ažuriramo brzine. 
- Svi parametri simulacije se učitavaju iz datoteke i na kraju simulacije upisuju u datoteku. Na taj način
je lako provjeriti korektnost programa. 
- Sile treba računati na dva načina. Prvi je direktan i dan je ovim pseudo-kodom:

```
for each particle i
    force[i] = 0;

for each particle i {
   for each particle j {
       if(i != j) {
            d = pos[i] − pos[j];
			d_norm = norm(d);
            force[i] -=  G ∗ mas[i] ∗ mas[j] / (d_norm^3) ∗ d;
       }
   }
}
```

Drugi pristup uzima u obzir zakon akcije i reakcije i na osnovu njega prepolavlja broj računanja sila.
Ovdje je pseudo-kod:

```
for each particle i
    force[i] = 0;

for each particle i {
   for each particle j > i {
            d = pos[i] − pos[j];
			d_norm = norm(d);
			forceIJ =  G ∗ mas[i] ∗ mas[j] / (d_norm^3) ∗ d;
            force[i] -=  forceIJ;
            force[j] +=  forceIJ;
   }
}
```

Oba pristupa moraju dati identična rješenja, ali predstavljaju razliku u paralelizaciji koda.


- Potrebno je razviti serijski kod i testirati ga na vlastitim primjerima. Pored toga dan je 
ulazna datoteka `nbody.input` za dodatno testiranje. 
- Funkcije za upis i ispis podataka su dane kako ne bi bilo razlika u čitanju i pisanju 
podataka. Učitavaju se svi parametri simulacije (uključujući i gravitacijsku konstantu) kako bi
simulacija bila posve kontrolirana ulaznom datotekom.
- Za razvoj grafičkog prikaza rezultata dobivaju se dodatni bodovi (50 %). 


## Literatura

- P.M. Visser, _Collision detection for N-body Kepler systems_, Astronomy & Astrophysics manuscript no. AA43754
December 2, 2022.
- Peter S. Pacheco:  _An Introduction to Parallel Programming_, Elsevier, 2011. 
