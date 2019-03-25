#ifndef SHELL_SORT_CPP
#define SHELL_SORT_CPP

#include <vector>
#include <iostream>
#include <thread>
#include <algorithm>

#include "util.cpp"

using std::vector;
using std::thread;
using std::cout;
using std::endl;

void shell_sort(vector<int>& vec, int low, int high) {
    int size = 1 + (high - low);
    for (int gap = size / 2; gap > 0; gap /= 2) {
        for (int i = low + gap; i < low + size; i += 1) {
            int temp = vec[i];
            int j;
            for (j = i; j >= low + gap && vec[low + j - gap] > temp; j -= gap) {
                vec[j] = vec[j - gap];
            }
            vec[j] = temp;
        }
    }
}

int serial_shell_sort(vector<int>& vec) {
    for (int gap = vec.size() / 2; gap > 0; gap /= 2) {
        for (int i = gap; i < vec.size(); i += 1) {
            int temp = vec[i];
            int j;
            for (j = i; j >= gap && vec[j - gap] > temp; j -= gap) {
                vec[j] = vec[j - gap];
            }
            vec[j] = temp;
        }
    }
    return 0;
}

void shell_worker(vector<int>& vec, int gap, int start) {
    int threads;
    if (gap > NUM_THREADS) {
        threads = NUM_THREADS;
    } else {
        threads = gap;
    }

    for (int i = start; i < vec.size(); i += threads) {
        int temp = vec[i];
        int j;
        for (j = i; j >= start && vec[j - gap] > temp; j -= gap) {
            vec[j] = vec[j - gap];
        }
        vec[j] = temp;
    }
}

void parallel_shell_sort(vector<int>& vec) {
    vector<thread> workers;
    int gap;
    for (gap = vec.size() / 2; gap > 1; gap /= 2) {
        workers.clear();
        for (int start = gap; start < gap + NUM_THREADS; start++) {
             workers.push_back(thread([start, gap, &vec]() {
                shell_worker(vec, gap, start);
             }));
        }
        std::for_each(workers.begin(), workers.end(), [](std::thread &worker) {
            worker.join();
        });
    }
    if (gap == 1) {
        serial_shell_sort(vec);
    }
}

#endif  // SHELL_SORT_CPP
