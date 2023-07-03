#pragma once 

#include <cmath>
#include <iostream>

// Točka u prostoru odnosno vektor.
struct Point{
    double x;
    double y;

    Point (double x = 0, double y = 0) 
        : x(x), y(y) {}

    Point& operator =(const Point& a) {
        x = a.x;
        y = a.y;
        return *this;
    } 

    Point operator +(const Point& a) const{
        return Point(x + a.x, y + a.y);
    }

    Point operator -(const Point& a) const{
        return Point(x - a.x, y - a.y);
    }

    Point operator*(double f) const{
        return Point(f * x, f * y);
    }

    Point operator/(double f) const{
        return Point(x / f, y / f);
    }
};

// Eventualne dodatne inline funkcije i operatori na tipu Point
inline double Norm(Point p) {
    return std::sqrt(p.x*p.x + p.y*p.y);
}

// Alociraj memoriju za sile/ubrzanja i inicijaliziraj ih nulama.
Point *  init_force(int N);

// Izračunaj silu na sva tijela. Prva (direktna) verzija.
void calc_force1(const double *mass, const Point * pos, Point * acc,  int N, double G, double limit);

// Izračunaj silu na sva tijela. Druga, optimizirana verzija.
void calc_force2(const double *mass, const Point * pos, Point * acc,  int N, double G, double limit);

// Pomakni sve točke.
void advance(double dt, Point * pos, const Point * vel, int N);

// Izračunaj nove brzine.
void velocity(double dt, Point * vel, const Point * acc, int N);

// Učitaj sve podatke iz datoteke.
void read_file(std::string const & filename
               , double ** mass
               //, float ** radius
               , Point ** pos
               , Point ** vel
               , int & N, double & dt
               , int & Nstep, double & G
               , double & limit);

// Ispiši sve podatke u datoteku. 
void write_file(std::ostream & out
               , const double * mass
               //, const float * radius
               , const Point * pos
               , const Point * vel
               , const Point * force
               , int N, double dt
               , int Nstep, double G
               , double limit);

