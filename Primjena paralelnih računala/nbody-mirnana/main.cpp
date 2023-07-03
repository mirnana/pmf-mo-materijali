#include <stdexcept>
#include <iostream>
#include <cmath>
#include <fstream>
#include <omp.h>

#include "nbody.h"

int main(int argc, char * argv[])
{
  double time = 0.0;

  for(int i = 0; i < 10; i++) {  
    double * mass = nullptr;       // mase 
    Point * pos   = nullptr;       // pozicije
    Point * vel   = nullptr;       // brzine
   
    int    N = 0;       // broj tijela
    double dt = 0.0;    // vremenski korak
    int    Nstep = 0;   // Broj koraka u simulaciji
    double G = 0;       // Gravitacijska konstanta
    double limit = 0.0; // limitator za silu
   
    std::string file_name("../nbody.input");
    if(argc > 1) 
   	  file_name = argv[1];
   
    read_file(file_name, &mass, &pos, &vel, N, dt, Nstep, G, limit);
   
    Point * acc = init_force(N); // sile/masa -- akceleracije   
   
    double begin = omp_get_wtime();
    for(int step=1; step<Nstep; ++step)
    {
        //calc_force1(mass, pos, acc, N, G, limit);
        calc_force2(mass, pos, acc, N, G, limit);
        advance(dt, pos, vel, N);
        velocity(dt, vel, acc, N);
    }
    double end = omp_get_wtime();
    
    std::ofstream out("nbody.results");
    if(!out)
      throw std::runtime_error("Cannot open nbody.results for writing.");
    write_file(out, mass, pos, vel, acc, N, dt, Nstep, G, limit);
    out.close();
   
    time += end - begin;
   
    delete [] acc;
    delete [] vel;
    delete [] pos;
    delete [] mass;
  
  }

  std::cout << "Prosječno vrijeme izvršavanja u 10 simulacija: " 
            << time/10 << "\n";
}
