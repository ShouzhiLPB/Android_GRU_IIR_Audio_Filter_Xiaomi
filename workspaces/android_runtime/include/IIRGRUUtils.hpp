#pragma once

#include <cmath>

template<typename T>
T normalize_frequency(const T Frequence, const T SampleRate)
{
    return  (log2(2 * Frequence) - log2(SampleRate)) / log2(SampleRate);
}