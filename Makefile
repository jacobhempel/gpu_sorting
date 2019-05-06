GCC = g++ -std=c++11 -pthread 

default: cpu-sorting gpu-sorting

run: cpu-sorting gpu-sorting
	./gpu-sorting

gpu-sorting: src/gpu_sorts.cu
	nvcc -O2 -g -o gpu-sorting src/gpu_sorts.cu

cpu-sorting: src/main.o
	$(GCC) -o cpu-sorting src/main.o    

src/main.o: src/main.cpp src/odd_even.o src/shell_sort.o src/quick_sort.o src/sample_sort.o src/util.o 
	$(GCC) -c src/main.cpp -o src/main.o 

src/odd_even.o: src/odd_even.cpp src/util.o
	$(GCC) -c src/odd_even.cpp -o src/odd_even.o 

src/shell_sort.o: src/shell_sort.cpp src/util.o
	$(GCC) -c src/shell_sort.cpp -o src/shell_sort.o 

src/quick_sort.o: src/quick_sort.cpp src/util.o src/shell_sort.o
	$(GCC) -c src/quick_sort.cpp -o src/quick_sort.o 

src/sample_sort.o: src/sample_sort.cpp src/quick_sort.cpp src/util.o 
	$(GCC) -c src/sample_sort.cpp -o src/sample_sort.o 

src/util.o: src/util.cpp
	$(GCC) -c src/util.cpp -o src/util.o 

clean:
	rm src/*.o cpu-sorting
