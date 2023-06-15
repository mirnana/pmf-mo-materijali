#pragma once
# include <mutex>
class Account{
    public:
        explicit Account(double);
        double getBalance() const;
        void deposit(double);
        void withdraw(double);
    private:
        double mBalance;
        mutable std::mutex acc_mutex; // mutable => smije se mijenjati i u const funkcijama
};