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
    // cout << "doing some sampling!" << endl;
    for (int i = 0; i < NUM_THREADS * 4; i++) {
        samples.push_back(vec[rand() % vec.size()]);
    }
    serial_quicksort(samples);
    for (int i = 1; i < NUM_THREADS; i++) {
        returns.push_back(samples[i * NUM_THREADS]);
    }
    // cout << "done sampling!!" << endl;
    return returns;
}

void sample_sort(vector<int>& vec) {
    vector<thread> workers;
    vector<vector<int>> buckets;

    auto splitters = sample(vec);

    int left = 0;
    int right = 0;

    for (int i = 0; i < NUM_THREADS; i++) {
        if (i < NUM_THREADS - 1) {
            right = splitters[i];
        } else {
             right = 210000000;
        }

        workers.push_back(thread([left, right, &buckets, &vec]() {
            vector<int> bucket;
            bucket.reserve(vec.size() / NUM_THREADS);
            sample_sort_worker(vec, bucket, left, right);
            buckets.push_back(bucket);
        }));

        left = right;
    }

    std::for_each(workers.begin(), workers.end(), [](std::thread &t) {
        t.join();
    });

    // cout << "Just need to sort the buckets now!" << endl;

    bool done = false;

    while(!done) {
        done = true;

        for (int i = 0; i < buckets.size() - 1; i++) {
            if (buckets[i][0] > buckets[i+1][0]) {
                auto temp = buckets[i];
                buckets[i] = buckets[i + 1];
                buckets[i + 1] = temp;
                done = false;
            }
        }

    }

    // cout << "Just need to concat now!" << endl;

    vector<int> sorted;
    for (auto bucket: buckets) {
        sorted.insert( sorted.end(), bucket.begin(), bucket.end() );
    }

    vec = sorted;
}

#endif  // SAMPLE_SORT_CPP