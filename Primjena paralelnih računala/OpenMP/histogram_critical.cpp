#include <iostream>
#include <random>
#include <cmath>

int main() {
    const int N = 10'000'000;
    double * data = new double[N];

    std::random_device rd;
    std::default_random_engine re;
    //re.seed(3456); // ovo [fiksni seed] je ok kad zelimo testirati jer re onda vraca uvijek isti btoj 
    re.seed(rd());

    std::normal_distribution<> normal(0.0, 1.0);

    for(int i = 0; i < N; i++) {
        data[i] = normal(re);
    }

    // hist[i] je broj el i (j-1,j] za j=i-4
    int hist[10];
    for(int i = 0; i < 10; i++) hist[i] = 0;
    
    #pragma omp parallel 
    {
        int hist_local[10];
        for(int i = 0; i < 10; i++) hist_local[i] = 0;

        #pragma omp for
        for(int i = 0; i < N; i++) {
            if (data[i] > -5 && data[i] <= 5) {
                int n = std::ceil(data[i]) + 4;
                hist_local[n]++;
            }
        }

        #pragma omp critical
        {
            for(int i = 0; i < 10; i++) 
                hist[i] += hist_local[i];
        }
    }

    for(int i = 0; i < 10; i++) {
        std::cout << "hist[" 
                  << i 
                  << "] = " 
                  << hist[i] 
                  << "\n";
    }

    return 0;
}