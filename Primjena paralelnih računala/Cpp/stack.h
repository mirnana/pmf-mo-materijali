#pragma once
#include <mutex>
#include <stack>
#include <memory>

template <typename T>
class Stack {
    public:
    Stack() = default;
    Stack(Stack const & ) = delete;   // ne zamaramo se s copy konstruktorom
    Stack & operator=(Stack const &) = delete; // ne zamaramo se s operatorom pridruživanja

    void push(T const &);
    void pop(T &);
    std::shared_ptr<T> pop();
    
    private:
    std::stack<T> mData;
    mutable std::mutex mMutex;
};

template<typename T>
void Stack<T>::push(T const & data) {
    std::lock_guard<std::mutex> lg(mMutex);
    mData.push(data);
}

template<typename T>
void Stack<T>::pop(T & data) {
    std::lock_guard<std::mutex> lg(mMutex);
    if(mData.empty()) {
        throw std::runtime_error("Empty stack!");
    }
    data = mData.top();
    mData.pop();
}

template<typename T>
std::shared_ptr<T> Stack<T>::pop() {
    std::lock_guard<std::mutex> lg(mMutex);
    if(mData.empty()) {
        throw std::runtime_error("Empty stack!");
    }
    std::shared_ptr<T> pt = std::make_shared<T>(mData.top());
    mData.pop();
    return pt;
}