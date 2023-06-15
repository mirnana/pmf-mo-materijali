#include <iostream>
#include <string>
#include <random>
#include <omp.h>

template <typename T>
int partition(T * a, int lo, int hi){
	// Odabir pivotnog elementa.
	int pivotIdx = (lo+hi)/2;
	T   pivotVal = a[pivotIdx];
	// Pivotni element ide na kraj. To je nužno za korektnost algoritma.
	std::swap(a[pivotIdx], a[hi]);
	int index = lo; // mjesto slobodno za smještaj manjeg elementa
	for(int i = lo; i<hi; ++i){
		if(a[i] < pivotVal){
			std::swap(a[i],a[index]);
			++index;
		}
	}
	std::swap(a[index], a[hi]); // pivot ide na svoje mjesto
	return index;
}

template <typename T>
void quicksort(T * a, int lo, int hi){
	if(lo < hi){
        int minsize=10000;

		int p = partition(a, lo, hi); // element a[p] je na svom mjestu

        #pragma omp task final(p-lo < minsize) \
                mergeable firstprivate(lo, p)
        quicksort(a, lo, p-1);

        #pragma omp task final(hi-p < minsize) \
                mergeable firstprivate(p, hi)
		quicksort(a, p+1, hi);
	}
}

template <typename T>
void print(T* a, int lo, int hi) {
    for(int i = lo; i < hi; i++) {
        std::cout << a[i] << ", ";
    }
    std::cout << a[hi] << "\n";
}

template <typename T>
bool test(T*a, int lo, int hi) {
    for(int i = lo; i < hi; i++) {
        if (a[i] > a[i+1])
            return false;
    }
    return true;
}

int main(int argc, char*argv[]) {

    std::random_device rd;
    std::default_random_engine re;
    std::uniform_int_distribution dist(10,100);
    re.seed(rd());

    int N = 100;
    if(argc > 1) 
        N = std::stoi(argv[1]);
    int * a = new int[N];
    for(int i = 0; i < N; i++) {
        a[i] = dist(re);
    }

    bool output = true;
    if(argc>2) output = false;

    double begin = omp_get_wtime();
    #pragma omp parallel 
    {
        #pragma omp single nowait
        quicksort(a, 0, N-1);
    }
    double end = omp_get_wtime();
    
    if(output) 
        print(a, 0, N-1);

    std::cout << std::boolalpha << test(a, 0, N-1) 
            << ", vrijeme: " << end-begin << "\n";

    return 0;
}