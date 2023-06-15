#include <iostream>
#include <random>
#include <omp.h>

// Provjeri je li niz sortiran u rastućem poretku.
// Niz je  a[lo],a[lo+1],...,a[hi].
template <typename T>
bool is_sorted(const T * a, int lo, int hi){
	for(int i = lo; i<hi; ++i){
           if(a[i] > a[i+1])
			   return false;
	}
    return true;
}

template <typename T>
void print(const T * a, int lo, int hi, const char * text = "a: "){
	std::cout << text;
	for(int i = lo; i<= hi; ++i)
		std::cout << a[i] <<",";
	 std::cout << "\n";
}


template <typename T>
void merge(T * a, int lo, int mid, int hi) 
{
	int len_1st = mid - lo + 1;
	int len_2nd = hi - mid;
   	T *first  = new T[len_1st];		// prvi podniz je a[lo:mid] pa je duljine mid - lo + 1
	T *second = new T[len_2nd];		// drugi podniz je a[mid+1:hi] pa je duljine hi - (mid+1) + 1

	// punim kopije podnizova
	for(int i = 0; i < len_1st; i++) {
		first[i] = a[lo + i];
	}
	for(int i = 0; i < len_2nd; i++) {
		second[i] = a[mid + 1 + i];
	}

	// kopiram iz podnizova natrag u a
	for(int i = lo; i <= hi; i++) {
		if(len_1st > 0) {
			if(len_2nd > 0) {
				if(first[0] <= second[0]) {
					a[i] = first[0];
					first = first + 1;
					len_1st--;
				}
				else {
					a[i] = second[0];
					second = second + 1;
					len_2nd--;
				}
			}
			else {
				a[i] = first[0];
				first = first + 1;
				len_1st--;
			}
		}
		else {
			return;
		}
	}
}

template <typename T>
void mergesort( T * a, int lo, int hi) // granice su uključive
{
	int minsize = 10000;

	if(lo == hi) return;
	int mid = (lo + hi) / 2;

	# pragma omp task shared(a) final(mid-lo < minsize) firstprivate(lo, mid)
	mergesort(a, lo, mid);
	
	# pragma omp task shared(a) final(mid-lo < minsize) firstprivate(mid, hi)
	mergesort(a, mid + 1, hi);

	# pragma omp taskwait

	merge(a, lo, mid, hi);
}

int main(int argc, char * argv[])
{
	bool do_output = false; // Za ispis samog niza
    if(argc < 2){
		std::cout << "Usage: " << argv[0] << " N [output]\n";
		return 1;
	}
	int N = std::atoi(argv[1]);
	std::cout << "# elements = " << N << "\n";
	if(argc > 2)
		do_output = true;

    std::random_device rd;
    std::default_random_engine r_engine; // Generator slučajnih brojeva
    r_engine.seed( rd() ); 
	std::uniform_int_distribution<> dist(10,100);

	double time_parallel = 0.0, time_serial = 0.0;
	double begin, end;
	int num_calls = 10;
	bool sorted = true;

	for(int i = 0; i < num_calls; i++) {

		begin = omp_get_wtime();
		int *a = new int[N];
		for(int i=0; i<N; ++i)
			a[i] = dist(r_engine);

		if(do_output)
		{
			std::cout << "Slučajni niz:\n";
			for(int i=0; i<N; ++i)
				std::cout << a[i] << " ";
			std::cout << "\n";
		}
		end = omp_get_wtime();
		time_serial += end-begin;		

		begin = omp_get_wtime();
		# pragma omp parallel
		{
			# pragma omp single
			mergesort(a, 0, N-1);
		}
		end = omp_get_wtime();

		sorted *= is_sorted(a, 0, N-1);
		time_parallel += end - begin;

		begin = omp_get_wtime();
		if(do_output){
			std::cout << "Poslije sortiranja:\n";
			for(int i=0; i<N; ++i)
				std::cout << a[i] << " ";
			std::cout << "\n";
		}
		delete [] a;
		end = omp_get_wtime();
		time_serial += end-begin;
	}

	time_parallel /= num_calls;
	time_serial /= num_calls;
	std::cout << "Sortiram " << num_calls << " nizova zaredom\n" 
			  << "Nizovi su sortirani: " << std::boolalpha << sorted << "\n"
			  << "Prosječno vrijeme izvršavanja paralelnog koda: " << time_parallel << "\n"
			  << "Prosječno vrijeme izvršavanja serijskog koda: " << time_serial << "\n";
	
	return 0;
}
