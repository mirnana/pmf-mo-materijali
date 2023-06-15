#include <thread>
#include <chrono>
#include "ts_map.h"

Map<std::string, int> smap(5);

void fill_map() {
	int num_succeeded = 0;
	for(int i = 0; i < 150; i++) {
		std::string key = std::to_string(i * 16);
		num_succeeded += smap.insert(key, i);
	}
	std::string out = "fill je uspio " + std::to_string(num_succeeded) + " puta\n";
	std::cout << out;
}

void reassign_map() {
	int num_succeeded = 0;
	for(int i = 0; i < 150; i++) {
		std::string key = std::to_string(i * 8);
		num_succeeded += smap.assign(key, i + 2);
	}
	std::string out = "reassign je uspio " + std::to_string(num_succeeded) + " puta\n";
	std::cout << out;
}

void delete_map() {
	int num_succeeded = 0;
	for(int i = 0; i < 150; i++) {
		std::string key = std::to_string(i * 4);
		num_succeeded += smap.remove(key);
	}
	std::string out = "delete je uspio " + std::to_string(num_succeeded) + " puta\n";
	std::cout << out;
}

void get_map() {
	// puštam nit da odspava kako bi se smap stigao napuniti
	std::this_thread::sleep_for(std::chrono::milliseconds(3));	

	int num_succeeded = 0;
	for(int i = 0; i < 5000; i++) {	// obavljam puno više get-ova nego insert-ova
		std::optional<int> val = smap.get(std::to_string(i));
		if(val.has_value()) num_succeeded++;
	}
	std::string out = "get je uspio " + std::to_string(num_succeeded) + " puta\n";
	std::cout << out;
}

int main(){

	std::thread t_fill(fill_map);
	std::thread t_reassign(reassign_map);
	std::thread t_delete(delete_map);
	std::thread t_get(get_map);
	
	t_fill.join();
	t_reassign.join();
	t_delete.join();
	t_get.join();

	std::cout << "Veličina mape nakon manipulacija: " << smap.size() << "\n\n";
	smap.print();
	
	return 0;
}
