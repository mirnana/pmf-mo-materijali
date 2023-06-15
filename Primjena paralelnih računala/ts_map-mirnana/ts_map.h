#pragma once

#include <functional>  // za hash
#include <optional>
#include <iostream>
#include <shared_mutex>
#include <list>
#include <utility>		// za pair strukturu

template <typename Key, typename Value, typename Hash = std::hash<Key>>
class Map{
	public:
		using key_type = Key;
		using mapped_type = Value;
		using hash_type = Hash;

		Map(int num_buckets = 5, Hash hash_ = Hash{});

		Map(Map const &) = delete;
		Map & operator=(Map const &) = delete;

        bool insert(Key const & key, Value const & val);
        bool assign(Key const & key, Value const & val);
		bool remove(Key const & key); 
		std::optional<Value> get(Key const & key) const; 
		void print() const;
		size_t size() const;
	private:
		int _size;
		mutable std::shared_mutex size_mutex;

		int _num_buckets;
		std::list<std::pair<Key, Value>> *buckets; // NE KORISTITI OBIČNA POLJA
		mutable std::shared_mutex *mutexes;
};

// ================= Metode klase Map =========================

template <typename Key, typename Value, typename Hash>
inline Map<Key, Value, Hash>::Map(int num_buckets, Hash hash) {
	_num_buckets = num_buckets;	
	buckets = new std::list<std::pair<Key, Value>>[num_buckets];
	mutexes = new std::shared_mutex[num_buckets];  // ZABORAVLJENA DEALOKACIJA! DESTRUKTOR!
	_size = 0;
}

template <typename Key, typename Value, typename Hash>
inline bool Map<Key, Value, Hash>::insert(Key const &key, Value const &val)
{
	int bucket = Hash{}(key) % _num_buckets;
	std::lock_guard<std::shared_mutex> lg(mutexes[bucket]);
	
	for(auto i = buckets[bucket].begin(); i != buckets[bucket].end(); i++) {
		if((*i).first == key) {
			return false;
		}
	}
	
	buckets[bucket].push_back(std::pair(key, val));

	std::lock_guard<std::shared_mutex> lg_size(size_mutex);
	_size++;

	return true;
}

template <typename Key, typename Value, typename Hash>
inline bool Map<Key, Value, Hash>::assign(Key const &key, Value const &val)
{
	int bucket = Hash{}(key) % _num_buckets;
	std::lock_guard<std::shared_mutex> lg(mutexes[bucket]);

	for(auto i = buckets[bucket].begin(); i != buckets[bucket].end(); i++) {
		if((*i).first == key) {
			(*i).second = val;
			return true;
		}
	}

	return false;
}

template <typename Key, typename Value, typename Hash>
inline bool Map<Key, Value, Hash>::remove(Key const &key)
{
	int bucket = Hash{}(key) % _num_buckets;
	std::lock_guard<std::shared_mutex> lg(mutexes[bucket]);
	
	int num_removed = buckets[bucket].remove_if(
		[this, key](std::pair<Key, Value> p){return key == p.first;}
		);
	if(num_removed > 0) {
		std::lock_guard<std::shared_mutex> lg_size(size_mutex);
		_size--;

		return true;
	}

    return false;
}

template <typename Key, typename Value, typename Hash>
inline std::optional<Value> Map<Key, Value, Hash>::get(Key const &key) const
{
	int bucket = Hash{}(key) % _num_buckets;
	std::shared_lock<std::shared_mutex> sl(mutexes[bucket]);

    std::optional<Value> ret_val;
    for (auto i = buckets[bucket].begin(); i != buckets[bucket].end(); i++)
    {
        if((*i).first == key) {
			ret_val = (*i).second;
		}
    }

    return ret_val;
}

template <typename Key, typename Value, typename Hash>
inline void Map<Key, Value, Hash>::print() const
{
    for (int i = 0; i < _num_buckets; i++) {
		std::shared_lock<std::shared_mutex> sl(mutexes[i]);
		std::string out = "Pretinac broj " + std::to_string(i) + ":\n";
		int c = 0;
		for (auto j = buckets[i].begin(); j != buckets[i].end(); ) {
			out += "(" + (*j).first + ", " + std::to_string((*j).second) + ")";
			if(++j != buckets[i].end()) {
				out += ", ";
				// radi preglednosti ispisujem 10 elemenata po retku
				if(++c % 10 == 0) out += "\n";	
			}
		}
		std::cout << out << "\n";
    }
}

template <typename Key, typename Value, typename Hash>
inline size_t Map<Key, Value, Hash>::size() const 
{
	std::shared_lock<std::shared_mutex> sl(size_mutex);
	return _size;
}
