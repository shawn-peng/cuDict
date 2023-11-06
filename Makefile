
libcuda_dict.so: cuda_dict.o
	$(CXX) -o $@ -shared $^

main: main.o libcuda_dict.so
	$(CXX) -o $@ $> -L . -lcuda_dict


