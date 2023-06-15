#include "account.h"
#include <iostream>
#include <thread>
#include <array>
#include <mutex>

//std::mutex acc_mutex; // selim ovo u klasu jer služi samo za pristup varijabli nBalance

// dodaj puno malih depozita.
void task(Account & acc){
    /*
    for(int i=1; i<=1000; ++i){
        acc_mutex.lock();
        acc.deposit(1.0);
        acc_mutex.unlock();
    }
    */

    // RAII princip
    //std::lock_guard<std::mutex> lg(acc_mutex); // pri kreiranju se lokot zaključava, a pri destrukciji s otključava. ako se baci exception, ovaj lokalni objekt ce biti unisten i time smo osigurali da ce lokot biti otkljucan!
    // ali ne zelimo zakljucati prije for petlje -- selim ga u metode koje se tiču mBalance-a
    for(int i=1; i<=1000; ++i){
        acc.deposit(1.0);
    }
}

int main(){
    Account acc(1000);
    
    // Kreiraj 5 threadova koji će dodavati depozite
    std::array<std::thread, 5> thrds;
    
    for(int i=0; i<5; ++i)
       thrds[i] = std::thread(task, std::ref(acc));

    for(int i=0; i<5; ++i)
       thrds[i].join();

    std::cout << acc.getBalance() << " (=6000)\n";
    return 0;
}