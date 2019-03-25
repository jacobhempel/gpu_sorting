#ifndef QUICK_SORT_CPP
#define QUICK_SORT_CPP

#include <vector>
#include <iostream>
#include <thread>
#include <algorithm>
#include <unistd.h>
#include <cmath>

#include "util.cpp"
#include "shell_sort.cpp"

using std::vector;
using std::thread;
using std::cout;
using std::endl;

const int MAX_DEPTH = ceil(log2(NUM_THREADS));

void serial_quicksort_worker(vector<int>& vec, int low, int high) {
    if (low < high) {
        int pivot = vec[high];    // pivot
        int i = (low - 1);  // Index of smaller element
        for (int j = low; j <= high- 1; j++) {
            if (vec[j] <= pivot) {
                i++;
                swap(vec, i, j);
            }
        }
        int part = i + 1;
        swap(vec, part, high);

        int part_neg = part - 1;
        int part_pos = part + 1;

        serial_quicksort_worker(vec, low, part_neg);
        serial_quicksort_worker(vec, part_pos, high);

    }
}

void parallel_quicksort_worker(vector<int>& vec, int low, int high, int depth) {
    if (low < high) {
        int pivot = vec[high];
        int i = (low - 1);
        for (int j = low; j <= high- 1; j++) {
            if (vec[j] <= pivot) {
                i++;
                swap(vec, i, j);
            }
        }

        int part = i + 1;
        swap(vec, part, high);

        int part_neg = part - 1;
        int part_pos = part + 1;

        if (depth <= MAX_DEPTH) {
            thread t1([low, part_neg, depth, &vec]() {
                parallel_quicksort_worker(vec, low, part_neg, depth + 1);
            });
            thread t2([part_pos, high, depth, &vec]() {
                parallel_quicksort_worker(vec, part_pos, high, depth + 1);
            });
            t1.join();
            t2.join();
        } else {
            serial_quicksort_worker(vec, low, part_neg);
            serial_quicksort_worker(vec, part_pos, high);
        }
    }
}

void serial_quicksort(vector<int>& vec) {
    serial_quicksort_worker(vec, 0, vec.size() - 1);
}

void parallel_quicksort(vector<int>& vec) {
    parallel_quicksort_worker(vec, 0, vec.size() - 1, 1);
}



#endif  // QUICK_SORT_CPP
