default: sorting

run: sorting
	./sorting

sorting: src/main.cpp src/odd_even.o src/shell_sort.o src/quick_sort.o src/util.o
	g++ src/main.cpp -o sorting -pthread

src/odd_even.o: src/odd_even.cpp src/util.o
	g++ -c src/odd_even.cpp -o src/odd_even.o -pthread

src/shell_sort.o: src/shell_sort.cpp src/util.o
	g++ -c src/shell_sort.cpp -o src/shell_sort.o -pthread

src/quick_sort.o: src/quick_sort.cpp src/util.o src/shell_sort.o
	g++ -c src/quick_sort.cpp -o src/quick_sort.o -pthread

src/util.o: src/util.cpp
	g++ -c src/util.cpp -o src/util.o -pthread

clean:
	rm src/*.o sorting
