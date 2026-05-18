#pragma once
#include <oboe/Oboe.h>

namespace oboe_utils {
    oboe::AudioStreamBuilder* create_base_builder(oboe::AudioStreamCallback* callback);
}
