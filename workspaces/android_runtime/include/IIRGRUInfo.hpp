#pragma once

#include <cstdint>

template<
    int64_t BatchSize,
    int64_t BufferSize,
    int64_t InputSize,
    int64_t HiddenSize,
    int64_t NumLayers>
struct IRRGRUInfo
{
    static constexpr int64_t batch_size  () { return BatchSize; }
    static constexpr int64_t buffer_size () { return BufferSize; }   
    static constexpr int64_t input_size  () { return InputSize; }    
    static constexpr int64_t hidden_size () { return HiddenSize; }   
    static constexpr int64_t num_layers  () { return NumLayers; }    
};

template<typename T>
concept IsIRRGRUInfo = requires(T t)
{
    std::bool_constant<(T::batch_size(),  true)>::value;
    std::bool_constant<(T::buffer_size(), true)>::value;
    std::bool_constant<(T::input_size(),  true)>::value;
    std::bool_constant<(T::hidden_size(), true)>::value;
    std::bool_constant<(T::num_layers(),  true)>::value;
};