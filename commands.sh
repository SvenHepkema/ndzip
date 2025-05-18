# Build benchmarks:
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native" -DNDZIP_BUILD_BENCHMARK=ON
cmake --build build -j

# Run Benchmarks:
./build/benchmark benchmark-data-description.csv -R 2 -a ndzip-cuda > results.out



# Compress and decompress
# 2905887 = n_bytes(file) / sizeof(double)
./build/compress -n 2905887 -i city_temperature_f.bin -o compressed-data.bin -t double -e cuda
./build/compress -d -n 2905887 -i compressed-data.bin -o decompressed-data.bin -t double -e cuda

#cmake -B build -DCMAKE_PREFIX_PATH='../hipSYCL-install/lib/cmake' -DHIPSYCL_PLATFORM=cuda -DCMAKE_CUDA_ARCHITECTURES=61 -DHIPSYCL_GPU_ARCH=sm_61 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-U__FLOAT128__ -U__SIZEOF_FLOAT128__ -march=native" -DNDZIP_BUILD_BENCHMARK=ON
