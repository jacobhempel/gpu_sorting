#ifndef SAMPLE_SORT_CPP
#define SAMPLE_SORT_CPP

#include <vector>
#include <iostream>
#include <thread>
#include <algorithm>
#include <unistd.h>
#include <cmath>
#include <cstdlib>

#include "util.cpp"
#include "quick_sort.cpp"

using std::vector;
using std::thread;
using std::cout;
using std::end;

struct bucket_compare {
    inline bool operator() (const vector<int>& vec1, const vector<int>& vec2) {
        return vec1[0] < vec2[0];
    }
};

void sample_sort_worker(const vector<int>& vec, vector<int>& bucket, int left, int right) {

    for (int i = 0; i < vec.size(); i++) {
        if (vec[i] >= left && vec[i] < right) {
            bucket.push_back(vec[i]);
        }
    }

    serial_quicksort(bucket);
}

vector<int> sample(const vector<int>& vec) {
    vector<int> samples;
    vector<int> returns;

    int interval = 32;
    // cout << "doing some sampling!" << endl;
    for (int i = 0; i < NUM_THREADS * interval; i++) {
        samples.push_back(vec[rand() % vec.size()]);
    }
    std::sort(samples.begin(), samples.end());
    for (int i = interval; i < samples.size(); i += interval) {
        returns.push_back(samples[i]);
    }

    std::sort(returns.begin(), returns.end());
    // cout << "done sampling!!" << endl;
    return returns;
}

void sample_sort(vector<int>& vec) {
    vector<thread> workers;
    vector<vector<int>> buckets;

    auto splitters = sample(vec);

    vector<int> empty;
    for (int i = 0; i < NUM_THREADS; i++) {
        buckets.push_back(empty);
    } 

    int left = 0;
    int right = 0;

    for (int i = 0; i < NUM_THREADS; i++) {
        if (i < NUM_THREADS - 1) {
            right = splitters[i];
        } else {
             right = 210000000;
        }

        workers.push_back(thread([left, right, i, &buckets, &vec]() {
            // vector<int> bucket;
            sample_sort_worker(vec, buckets[i], left, right);
            // buckets.push_back(bucket);
        }));

        left = right;
    }

    std::for_each(workers.begin(), workers.end(), [](std::thread &t) {
        t.join();
    });

    // cout << "Just need to sort the buckets now!" << endl;

    std::sort(buckets.begin(), buckets.end(), bucket_compare());

    vector<int> sorted;
    for (auto bucket: buckets) {
        sorted.insert( sorted.end(), bucket.begin(), bucket.end() );
    }

    vec = sorted;
}

#endif  // SAMPLE_SORT_CPP