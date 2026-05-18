cmake_minimum_required(VERSION 3.22.1)
project(gru_filter LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

get_filename_component(REPO_ROOT "${CMAKE_SOURCE_DIR}/../../../../../../" ABSOLUTE)

set(OBOE_SRC "${REPO_ROOT}/artifacts/build/arm64-v8a/_deps/oboe-src")
set(ORT_ROOT "${REPO_ROOT}/artifacts/build/arm64-v8a/_deps/onnxruntime-src")
set(ORT_HEADERS "${ORT_ROOT}/headers")
set(ORT_SO "${ORT_ROOT}/jni/arm64-v8a/libonnxruntime.so")

if(EXISTS "${REPO_ROOT}/oboe_boilerplates/acquire_audio_stream.cpp")
    set(OBOE_BOILERPLATES "${REPO_ROOT}/oboe_boilerplates")
else()
    set(OBOE_BOILERPLATES "${REPO_ROOT}/workspaces/oboe_boilerplates")
endif()

set(ANDROID_RUNTIME_INCLUDE "${REPO_ROOT}/workspaces/android_runtime/include")

add_subdirectory("${OBOE_SRC}" "${CMAKE_BINARY_DIR}/oboe_build")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_SO}"
)

add_library(gru_filter SHARED
    jni_bridge.cpp
    "${OBOE_BOILERPLATES}/acquire_audio_stream.cpp"
)

target_include_directories(gru_filter PRIVATE
    "${OBOE_BOILERPLATES}"           # first: acquire_audio_stream.hpp must match .cpp
    "${ANDROID_RUNTIME_INCLUDE}"
    "${ORT_HEADERS}"
)

target_link_libraries(gru_filter
    PRIVATE oboe
    PRIVATE onnxruntime
    PRIVATE log
    PRIVATE android
)
