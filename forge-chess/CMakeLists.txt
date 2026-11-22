cmake_minimum_required(VERSION 3.16)
project(forge-chess VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Build type ----
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- Options ----
option(FORGE_NATIVE "Hyper-optimize for THIS CPU (-march=native, best on your i5-1335U)" ON)
option(FORGE_PORTABLE "Portable binaries (no -march=native, still AVX2 where available)" OFF)
option(FORGE_LTO "Enable link-time optimization" ON)

# ---- Common flags ----
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -funroll-loops -fomit-frame-pointer -fno-math-errno -fno-trapping-math)
  if(FORGE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_err)
    if(_ipo_ok)
      set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
  endif()
endif()

# ---- CPU tuning ----
# FORGE_NATIVE=ON  -> -march=native -mtune=native  (MAX on your current machine)
# FORGE_NATIVE=OFF -> portable baseline + AVX2/BMI2 if compiler supports it
if(FORGE_NATIVE AND NOT FORGE_PORTABLE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" _has_march_native)
  if(_has_march_native)
    add_compile_options(-march=native -mtune=native)
    add_compile_definitions(FORGE_NATIVE_BUILD=1)
  endif()
else()
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mavx2" _has_avx2)
  check_cxx_compiler_flag("-mbmi2" _has_bmi2)
  if(_has_avx2)
    add_compile_options(-mavx2)
    add_compile_definitions(FORGE_HAS_AVX2=1)
  endif()
  check_cxx_compiler_flag("-mfma" _has_fma)
  if(_has_fma)
    add_compile_options(-mfma)
  endif()
  if(_has_bmi2)
    add_compile_options(-mbmi2)
  endif()
  # Baseline portable: at least SSE4.1 + popcnt where available
  check_cxx_compiler_flag("-mpopcnt" _has_popcnt)
  if(_has_popcnt)
    add_compile_options(-mpopcnt)
  endif()
endif()

# Force AVX2 define if compiler already targets it (e.g. -march=native on AVX2 CPU)
# The sources also do runtime-safe scalar fallback, so this is always safe.
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("#ifndef __AVX2__\n#error no avx2\n#endif\nint main(){}" _compiler_has_avx2)
if(_compiler_has_avx2)
  add_compile_definitions(FORGE_HAS_AVX2=1)
endif()

# Threads
find_package(Threads REQUIRED)

# ---- Engine library ----
add_library(forge_lib STATIC
  src/position.cpp
  src/nnue.cpp
  src/search.cpp
  src/uci.cpp
  src/trainer.cpp
)
target_include_directories(forge_lib PUBLIC src)
target_link_libraries(forge_lib PUBLIC Threads::Threads)

# ---- Binaries ----
add_executable(forge src/engine_main.cpp)
target_link_libraries(forge PRIVATE forge_lib)

add_executable(forge-train src/trainer_main.cpp)
target_link_libraries(forge-train PRIVATE forge_lib)

# Perft / sanity test binary (optional, tiny)
add_executable(forge-perft src/perft_main.cpp)
target_link_libraries(forge-perft PRIVATE forge_lib)

# ---- Install (any OS with a C++ compiler) ----
install(TARGETS forge forge-train RUNTIME DESTINATION bin)

message(STATUS "Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "FORGE_NATIVE    : ${FORGE_NATIVE}")
message(STATUS "FORGE_PORTABLE  : ${FORGE_PORTABLE}")
message(STATUS "C++ compiler    : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
