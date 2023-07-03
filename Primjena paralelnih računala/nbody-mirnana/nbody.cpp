#include "nbody.h"
#include <stdexcept>
#include <random>
#include <cassert>
#include <fstream>
#include <sstream>

// Alociraj memoriju za sile/ubrzanja i inicijaliziraj ih nulama.
Point *  init_force(int N) {
    Point * list = new Point[N];
    for(int i = 0; i < N; i++) {
        list[i].x = list[i].y = 0.0;
    }
    return list;
}

// Izračunaj silu/akceleraciju na sva tijela.  Prva verzija (direktna).
void calc_force1( const double *mass
                , const Point * pos
                , Point * acc
                , int N
                , double G
                , double limiter) {
    # pragma omp parallel for
    for(int i = 0; i < N; i++) { 
        acc[i].x = acc[i].y = 0.0; 
    }

    # pragma omp parallel for
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            if (i != j) {
                Point d = pos[i] - pos[j]; 
                double d_norm = Norm(d) + limiter;

                acc[i] = acc[i] - d * (G * mass[j] / std::pow(d_norm, 3));
            }
        }
    }
}

// Izračunaj silu/akceleraciju na sva tijela. Druga verzija (optimizirana).
void calc_force2( const double *mass
                , const Point * pos
                , Point * acc
                , int N
                , double G
                , double limiter) {
    # pragma omp parallel for
    for(int i = 0; i < N; i++) { 
        acc[i].x = acc[i].y = 0.0; 
    }

    Point * threadAcc;

    # pragma omp parallel private(threadAcc) shared(acc)
    {
        threadAcc = init_force(N);

        # pragma omp for schedule(dynamic)
        for(int i = 0; i < N; i++) {
            for(int j = i + 1; j < N; j++) {
                Point d = pos[i] - pos[j];
                double d_norm = Norm(d) + limiter;
                
                Point accIJ = d * (G * mass[i] * mass[j] / std::pow(d_norm, 3));
                threadAcc[i] = threadAcc[i] - accIJ / mass[i];
                threadAcc[j] = threadAcc[j] + accIJ / mass[j];
            }
        }
// OVO JE SPORO, POSTOJI MOGUĆNOST ZA PARALELIZACIJU 
        for(int i = 0; i < N; i++) {
            # pragma omp atomic
            acc[i].x = acc[i].x + threadAcc[i].x;
            # pragma omp atomic
            acc[i].y = acc[i].y + threadAcc[i].y;
        }
    }
}

// Pomakni sve točke.
void advance( double dt
            , Point * pos
            , const Point * vel
            , int N) {
    for(int i = 0; i < N; i++) {
        pos[i] = pos[i] + vel[i] * dt;
    }
}

// Izračunaj nove brzine.
void velocity(double dt
            , Point * vel
            , const Point * acc
            , int N) {
    for(int i = 0; i < N; i++) {
        vel[i] = vel[i] + acc[i] * dt;
    }
}

// Rutine za čitanje i pisanje su zadane.
template <typename T>
void read(std::ifstream & in, std::string & line, T & t){
  std::getline(in, line); 
  while(line[0] == '#') 
		std::getline(in, line); 
  std::istringstream iss(line);
  iss >> t;
}

void read_file(std::string const & filename, 
		       double ** mass, Point ** pos, Point ** vel, 
		       int & N, double & dt, int & Nstep, double & G, double & limit)
{
  std::ifstream in(filename);
  if(!in)
    throw std::runtime_error("Cannot open " + filename + " for reading.");

  std::string line;
//  in >> std::ws;

  read<int>(in, line, N);
  read<double>(in, line, dt);
  read<int>(in, line, Nstep);
  read<double>(in, line, G);
  read<double>(in, line, limit);  // limit za slučaj bez kolicizije
  
  *mass = new double [N];
  if(!*mass)
    throw std::runtime_error("Alloc error 1.");

  *pos = new Point[N];
  if(!*pos)
    throw std::runtime_error("Alloc error 3.");

  *vel = new Point[N];
  if(!*vel)
    throw std::runtime_error("Alloc error 4.");

  std::getline(in, line); 
  while(line[0] == '#') 
		std::getline(in, line);

  int i=0;
  do{
    std::istringstream  iss(line);
    iss >> (*mass)[i] >> (*pos)[i].x >> (*pos)[i].y >> (*vel)[i].x >> (*vel)[i].y;
    ++i;
  }
  while(std::getline(in, line));
  assert(i == N);

  in.close();
}

void write_file(std::ostream & out, const double * mass, const Point * pos, 
               const Point * vel, const Point * force, int N, double dt, int Nstep, double G, double limit)
{
 
  out << N     << "  # N\n";
  out << dt    << "  # dt\n";
  out << Nstep << "  # Nstep\n";
  out << G     << "  # G\n";
  out << limit << "  # Limit\n";
  
  int i=0;
  while(i <N){
    out << "m = " << mass[i] << ", pos=(" << pos[i].x << "," << pos[i].y 
		<< "), vel=(" << vel[i].x << "," << vel[i].y <<"),"
		<< " force=(" << force[i].x << "," << force[i].y <<  ")\n";
    ++i;
  }  
}


