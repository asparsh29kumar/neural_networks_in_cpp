# neural_networks_in_cpp
The repository contains implementation of a two layer neural neural in c++ from scratch.It also contains an implementation of the same in mpi.

To run two_layer_net.cpp
g++ two_layer_net_mpi.cpp -std=c++11 -o two_layer_net
./two_layer_net

To run two_layer_net_mpi.cpp
mpic++ two_layer_net_mpi.cpp -std=c++11 -o two_layer_net
mpiexec -n 20 ./two_layer_net
