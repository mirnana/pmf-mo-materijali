#include <iostream>

int main()
{
#pragma omp parallel
	{

#pragma omp single
		{
			std::cout << "Danas pada ";
#pragma omp task
			{    // task #1
				std::cout << "snijeg ";
			}
#pragma omp task
			{   // task #2
				std::cout << "vlažan ";
			}
#pragma omp task
			{   // task #3
				std::cout << "i ";
			}
#pragma omp task
			{   // task #4
				std::cout << "obilan ";
			}

#pragma omp taskwait
            std::cout << "!\n";
		} // kraj single područja

	} // kraj paralelnog područja
	return 0;
}