#include <iostream>
#include <thread>     // za std::thread

void hello() {
  std::cout << "Message form hello().\n";
}

int main() {
  std::thread t(hello); // kreiranje dodatne programske niti

  std::cout << "Message from main().\n";

  t.join();
  return 0;
}