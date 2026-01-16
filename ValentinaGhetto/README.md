Run with

g++ -o main -I/opt/homebrew/include -I/opt/homebrew/Cellar/root/6.36.06_1/include/root \
    -I/Users/glucia/Projects/CATS/DLM_glucia/install/include \
    -L/opt/homebrew/lib -L/opt/homebrew/Cellar/root/6.36.06_1/lib -L/Users/glucia/Projects/CATS/DLM_glucia/install/CMake \
    -Wl,-rpath,/Users/glucia/Projects/CATS/DLM_glucia/install/CMake \
    main.cpp MiracleCoulomb.cpp \
    -lCATS -lgsl -lgslcblas -lflint \
    $(root-config --cflags --libs)

./main