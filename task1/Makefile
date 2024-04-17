all: sin_sum_double sin_sum_float

sin_sum_double: main.cpp
	g++ -o $@ $< -DUSE_DOUBLE

sin_sum_float: main.cpp
	g++ -o $@ $< -DUSE_FLOAT

clean:
	rm -f sin_sum_double sin_sum_float
