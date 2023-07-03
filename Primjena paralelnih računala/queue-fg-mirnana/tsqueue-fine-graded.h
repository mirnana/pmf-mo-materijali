#pragma once

#include <utility>
#include <memory>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>

template <typename T>
class Queue{
    struct Node{
        T  data;  
        Node * next;
    };

    mutable std::mutex head_mutex;
    Node * head;

    mutable std::mutex tail_mutex;
    Node * tail;

    std::condition_variable cv;
    
    public:
        Queue() : head(new Node), tail(head) {}
        ~Queue();

        Queue(Queue const &) = delete;
        Queue & operator=(Queue const &) = delete;

        bool try_pop(T &);
        void wait_pop(T &);
        void push(T);
		bool empty() const;

        void print();
};

template <typename T>
bool Queue<T>::empty() const{
	std::lock_guard<std::mutex> lg_head(head_mutex);
    std::lock_guard<std::mutex> lg_tail(tail_mutex);

    if(head == tail) return true;
    return false;
}

template <typename T>
bool Queue<T>::try_pop(T & t){
    if(this->empty())
        return false;
    else {
	    std::lock_guard<std::mutex> lg_head(head_mutex);
        t = head->data;
        head = head->next;
		// DEALOKACIJA?
        return true;
    }        
}

template <typename T>
void Queue<T>::wait_pop(T & t){
	std::unique_lock<std::mutex> ul_head(head_mutex);
    cv.wait(ul_head, [this]{
        std::lock_guard<std::mutex> lg_tail(tail_mutex);
        return head != tail;
    });
    t = head->data;
    head = head->next;
	// DEALOKACIJA?
}

template <typename T>
void Queue<T>::push(T t){
	std::lock_guard<std::mutex> lg_tail(tail_mutex);
    Node *new_node = new Node;  // ISPRED ZAKLJUČAVANJA
    tail->data = t;
    tail->next = new_node;
    tail = new_node;
    cv.notify_all();
}

template <typename T>
Queue<T>::~Queue(){
	// ZAKLJUČAVANJE
	Node *i = head;
    while (i != tail) {
        Node *j = i->next;
        delete i;
        i = j;
    }
    delete tail;
}

template <typename T>
void Queue<T>::print(){
    std::lock_guard<std::mutex> lg_head(head_mutex);
    std::lock_guard<std::mutex> lg_tail(tail_mutex);

    Node *i = head;
    std::string out = "Queue: ";
    while(i != tail) {
        out +=  std::to_string(i->data) + " <-- ";
        i = i->next;
    }
    out += "|\n";
    std::cout << out;
}
