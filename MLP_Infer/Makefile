CC=g++
IDIR_FRUGAL=./libs/frugally-deep-master/include
IDIR_FUNC=./libs/FunctionalPlus-master/include
IDIR_EIGEN=./libs/eigen
IDIR_JSON=./libs/json/include
CFLAGS=-I$(IDIR_FUNC) -I$(IDIR_FRUGAL) -I$(IDIR_EIGEN) -I$(IDIR_JSON) -O3 -march=native 

# find_package(frugally-deep CONFIG REQUIRED)
# target_link_libraries(main PRIVATE frugally-deep::fdeep)

hellomake: mlp_inference.cpp
	$(CC) $(CFLAGS) -o mlp_inference mlp_inference.cpp

test:
	./mlp_inference # python3 convert_model.py my_model_5_2048.h5 fdeep_model.json
