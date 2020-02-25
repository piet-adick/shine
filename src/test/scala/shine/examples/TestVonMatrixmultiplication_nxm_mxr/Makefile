#g++ -Wall -o hello hello.cpp -std=c++14
C++FLAGS := -Wextra -pedantic -std=c++14 -fPIC
aufgabe1: aufgabe1.o
	g++ $(C++FLAGS) -o aufgabe1 aufgabe1.o
#g++ -Wall -o prog main.o -std=c++14

aufgabe1.o: aufgabe1.cpp
	g++ $(C++FLAGS) -c aufgabe1.cpp

clean:
	rm *.o aufgabe1
