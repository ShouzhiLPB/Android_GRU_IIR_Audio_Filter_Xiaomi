include(FetchContent)

# ── Configuration ──────────────────────────────────────────────────────────────
set(ONXX_RT_URL      "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.24.2/onnxruntime-android-armeabi-v7a-static_lib-1.24.2.zip")
set(ONXX_RT_SHA256   "429b223c9294488459e28877df4d5229c04cb1489be14865017e06fd86f3cd50") # Replace with actual SHA-256 hash
set(ONXX_RT_NAME     "onnxruntime")     # Name used internally by FetchContent

# ── Download & extract ─────────────────────────────────────────────────────────
FetchContent_Declare(
    ${ONXX_RT_NAME}
    URL        ${ONXX_RT_URL}
    URL_HASH   SHA256=${ONXX_RT_SHA256}
    DOWNLOAD_NO_EXTRACT FALSE            # Ensure the archive is extracted
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(${ONXX_RT_NAME})

# ── Locate the .a file inside the extracted folder ─────────────────────────────
# FetchContent extracts into: <build_dir>/_deps/mylib-src/
# Adjust the sub-path to match the actual layout of your archive.
set(ONXX_RT_SRC_DIR  "${${ONXX_RT_NAME}_SOURCE_DIR}")   # e.g. _deps/mylib-src

# ── Create an IMPORTED target so the rest of the build uses it cleanly ─────────
if(NOT TARGET Ort::Ort)
    add_library(Ort::Ort STATIC IMPORTED GLOBAL)
    set_target_properties(Ort::Ort PROPERTIES
        IMPORTED_LOCATION             "${ONXX_RT_SRC_DIR}/lib/libonnxruntime.a"
        INTERFACE_INCLUDE_DIRECTORIES "${ONXX_RT_SRC_DIR}/include"
    )
endif()

message(STATUS "Ort static library : ${ONXX_RT_SRC_DIR}")
message(STATUS "Ort include dir    : ${ONXX_RT_SRC_DIR}/include")