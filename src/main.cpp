#include <cstdlib>
#include <iostream>
#include <vector>
#include <thread>
#include <ctime>
#include <cmath>

#include "odd_even.cpp"
#include "util.cpp"
#include "shell_sort.cpp"
#include "quick_sort.cpp"
#include "sample_sort.cpp"
#include "odd_even.cu"

using std::vector;
using std::thread;
using std::cout;
using std::endl;


// HELPER FUNCTIONS TO RUN SORTS
void do_serial_bubble_sort(vector<int> vec) {
    cout << "SERIAL   - bubble sort   => ";
    auto t1 = std::chrono::high_resolution_clock::now();
    bubble_sort(vec);
    auto t2 = std::chrono::high_resolution_clock::now();
    if (is_sorted(vec)) {
        cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
             << " milliseconds\n";
    } else {
        cout << "FAILED TO PRODUCE A SORTED LIST" << endl;
    }
}

void do_parallel_odd_even_sort(vector<int> vec) {
    auto t1 = std::chrono::high_resolution_clock::now();
    odd_even_sort(vec);
    auto t2 = std::chrono::high_resolution_clock::now();
    if (is_sorted(vec)) {
        std::cout << "PARALLEL - odd-even sort => "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                  << " milliseconds\n";
    } else {
        cout << "ODD-EVEN SORT FAILED TO PRODUCE A SORTED LIST" << endl;
    }
}

void do_GPU_parallel_odd_even_sort(vector<int> vec) {
    int* array;
    int size = as_heap_array(vec, array);
    auto t1 = std::chrono::high_resolution_clock::now();
    GPU_odd_even_sort(vec);
    auto t2 = std::chrono::high_resolution_clock::now();
    if (is_sorted(vec)) {
        std::cout << "PARALLEL - odd-even sort => "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                  << " milliseconds\n";
    } else {
        cout << "GPU ODD-EVEN SORT FAILED TO PRODUCE A SORTED LIST" << endl;
    }
    delete array;
}

void do_serial_shell_sort(vector<int> vec) {
    auto t1 = std::chrono::high_resolution_clock::now();
    serial_shell_sort(vec);
    auto t2 = std::chrono::high_resolution_clock::now();
    if (is_sorted(vec)) {
        std::cout << "SERIAL   - shell sort    => "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                  << " milliseconds\n";
    } else {
        cout << "SHELL SORT FAILED TO PRODUCE A SORTED LIST" << endl;
    }
}

void do_parallel_shell_sort(vector<int> vec) {
    auto t1 = std::chrono::high_resolution_clock::now();
    parallel_shell_sort(vec);
    auto t2 = std::chrono::high_resolution_clock::now();
    if (is_sorted(vec)) {
        std::cout << "PARALLEL - shell sort    => "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                  << " milliseconds\n";
    } else {
        cout << "PARALLEL SHELL SORT FAILED TO PRODUCE A SORTED LIST" << endl;
    }
}

void do_serial_quick_sort(vector<int> vec) {
    auto t1 = std::chrono::high_resolution_clock::now();
    serial_quicksort(vec);
    auto t2 = std::chrono::high_resolution_clock::now();
    if (is_sorted(vec)) {
        std::cout << "SERIAL   - quick sort    => "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                  << " milliseconds\n";
    } else {
        cout << "SERIAL QUICK SORT FAILED TO PRODUCE A SORTED LIST" << endl;
    }
}

void do_parallel_quick_sort(vector<int> vec) {
    auto t1 = std::chrono::high_resolution_clock::now();
    parallel_quicksort(vec);
    auto t2 = std::chrono::high_resolution_clock::now();
    if (is_sorted(vec)) {
        std::cout << "PARALLEL - quick sort    => "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                  << " milliseconds\n";
    } else {
        cout << "PARALLEL QUICK SORT FAILED TO PRODUCE A SORTED LIST" << endl;
    }
}

void do_sample_sort(vector<int> vec) {
    auto t1 = std::chrono::high_resolution_clock::now();
    sample_sort(vec);
    auto t2 = std::chrono::high_resolution_clock::now();
    if (is_sorted(vec)) {
        std::cout << "PARALLEL - sample sort   => "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
                  << " milliseconds\n";
    } else {
        cout << "SAMPLE SORT FAILED TO PRODUCE A SORTED LIST" << endl;
    }
}




int main(int argc, char const *argv[]) {
    srand(time(NULL));
    vector<int> vec;

    const int VEC_SIZE = 10000000;

    for (int i = 0; i < VEC_SIZE; i++) {
        vec.push_back(i);
    }

    cout << "sorting " << VEC_SIZE << " elements:" << endl;

    cout << "hardware concurrency = " << NUM_THREADS << endl;
    cout << "max depth            = " << MAX_DEPTH << endl;

    for (int i = 0; i < 5; i++) {
        shuffle(vec);
        // do_serial_bubble_sort(vec);
        // do_parallel_odd_even_sort(vec);

        do_serial_shell_sort(vec);
        do_parallel_shell_sort(vec);

        do_serial_quick_sort(vec);
        do_parallel_quick_sort(vec);

        do_sample_sort(vec);

        cout << "------------------------------------------------------------------------------" << endl;
    }

    return 0;
}
