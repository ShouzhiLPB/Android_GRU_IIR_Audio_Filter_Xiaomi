include(FetchContent)

FetchContent_Declare(
    oboe
    GIT_REPOSITORY https://github.com/google/oboe
    GIT_TAG 1.9.0 # crash happens with 1.10.0
)

FetchContent_MakeAvailable(oboe)