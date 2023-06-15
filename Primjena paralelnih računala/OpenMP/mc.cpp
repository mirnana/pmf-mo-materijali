#include <iostream>
#include <random>
#include <cmath>

int main() {
    int N = 0;
    std::cout << "unesi: \n";
    std::cin >> N;

    int circ_pts = 0;

    //double * data = new double[N];

    std::random_device rd;
    std::default_random_engine re1;
    std::default_random_engine re2;
    re1.seed(rd());
    re2.seed(rd());

    std::uniform_real_distribution<double> dist(-1,1);

    #pragma omp parallel for reduction(+: circ_pts)
    for(int i = 0; i < N; i++) {
        double x = dist(re1);
        double y = dist(re2);
        double d = x*x + y*y;
        
        if(d <= 1) circ_pts++;

    }

    double pi_approx = 4*static_cast<double>(circ_pts) / N;
    std::cout << "aproksimacija: " << pi_approx << "\n";
    std::cout << "Pi = " << M_PI << "\n";

    return 0;
}