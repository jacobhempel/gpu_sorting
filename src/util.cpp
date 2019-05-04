#ifndef UTIL_CPP
#define UTIL_CPP

#include <iostream>
#include <vector>
#include <thread>
#include <cstdlib>
#include <cmath>

using std::vector;
using std::thread;
using std::cout;
using std::endl;

const int NUM_THREADS = std::thread::hardware_concurrency();
const int MAX_DEPTH = ceil(log2(NUM_THREADS));

void swap(vector<int>& vec, int& a, int b) {
    int temp = vec[a];
    vec[a] = vec[b];
    vec[b] = temp;
}

bool is_sorted(vector<int> vec) {
    for (int i = 0; i < vec.size() - 1; i++) {
        if (vec[i] > vec[i + 1]) {
            return false;
        }
    }
    return true;
}

bool is_sorted(int* array, int size) {
    for (int i = 0; i < size - 1; i++) {
        if (array[i] > array[i+1]) {
            return false;
        }
    }
    return true;
}

void shuffle(vector<int>& vec) {
    for (int i = 0; i < vec.size(); i++) {
        swap(vec, i, rand() % vec.size());
    }
}

void print_vec(vector<int> vec) {
    for (int i = 0; i < vec.size(); i++) {
        cout << vec[i] << ", ";
    }
    cout << endl;
}

#endif  // UTIL_CPP
