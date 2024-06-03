pgc_parallel:
	pgc++ task.cpp -lboost_program_options -acc=multicore -Minfo=all -o task -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include/
pgc_gpu:
	pgc++ task.cpp -lboost_program_options -ta=tesla,managed -Minfo=all -o task -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include/
pgc_sequantial:
	pgc++ task.cpp -lboost_program_options -Minfo=all -o task -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include/
g++:
	g++ task.cpp -lboost_program_options -o task