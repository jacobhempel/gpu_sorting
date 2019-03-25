#ifndef ODD_EVEN
#define ODD_EVEN

#include <vector>
#include <iostream>
#include <thread>
#include <algorithm>

#include "util.cpp"

using std::vector;
using std::thread;
using std::cout;
using std::endl;


bool is_done(vector<bool> done) {
    for(int i = 0; i < done.size(); i++) {
        if (done[i] == false) {
            return false;
        }
    }
    return true;
}

bool odd_even_worker(vector<int>& vec, int start) {
    bool done = true;
    for (int i = start; i < vec.size() - 1; i += (NUM_THREADS * 2)) {
        if (vec[i] > vec[i + 1]) {
            swap(vec, i, i + 1);
            done = false;
        }
    }
    return done;
}

void odd_even_sort(vector<int>& vec) {
    vector<thread> workers;
    vector<bool> done;
    done.push_back(false);

    while (!is_done(done)) {
        workers.clear();
        done.clear();
        for(int j = 0; j < (NUM_THREADS * 2); j += 2) {
            workers.push_back(thread([j, &vec, &done]() {
                done.push_back(odd_even_worker(vec, j));
            }));
        }
        std::for_each(workers.begin(), workers.end(), [](std::thread &t) {
            t.join();
        });

        workers.clear();

        for(int j = 1; j < (NUM_THREADS * 2); j += 2) {
            workers.push_back(thread([j, &vec, &done]() {
                done.push_back(odd_even_worker(vec, j));
            }));
        }
        std::for_each(workers.begin(), workers.end(), [](std::thread &t) {
            t.join();
        });
    }
}

void bubble_sort(vector<int>& vec) {
    bool done = false;
    while (!done) {
        done = true;
        for (int i = 0; i < vec.size() - 1; i++) {
            if (vec[i] > vec[i + 1]) {
                swap(vec, i, i + 1);
                done = false;
            }
        }
    }
}

#endif
