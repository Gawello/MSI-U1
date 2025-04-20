# Kompilator
CXX = g++
CXXFLAGS = -std=c++17 -Wall -I./metoda -I./problem

# Pliki źródłowe i wynikowe
SRC = sterowanie/main.cpp metoda/classifier.cpp
OBJ = $(SRC:.cpp=.o)
EXEC = program

# Domyślna reguła
all: $(EXEC)

# Budowanie programu
$(EXEC): $(OBJ)
	$(CXX) $(OBJ) -o $@

# Kompilacja .cpp do .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Usuwanie plików tymczasowych
clean:
	rm -f $(OBJ) $(EXEC)
