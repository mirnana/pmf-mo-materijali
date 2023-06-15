#include "account.h"
#include <iostream>
#include <mutex>

Account::Account(double balance) : mBalance(balance) {}

double Account::getBalance() const { 
    std::lock_guard<std::mutex> lg(acc_mutex);
    return  mBalance; 
}

void Account::deposit(double amount){
    std::lock_guard<std::mutex> lg(acc_mutex);
    mBalance += amount;
}

void Account::withdraw(double amount) {
    std::lock_guard<std::mutex> lg(acc_mutex);
    if(mBalance < amount){
         std::cout << "Insufficient balance, withdraw denied." << std::endl;
         return;
    }
    mBalance -= amount;
}