//
// Created by andrei.pata on 7/24/19.
//

#pragma once

#include <string>

#ifdef USE_NVTX
#include <nvToolsExt.h>
#endif // USE_NVTX

namespace nie {
namespace nvtx {

/**
 * @brief  Colors for use with NVTX ranges.
 */
enum class Color : uint32_t {
    // @formatter:off
    kGreen     = 0xff00ff00,
    kBlue      = 0xff0000ff,
    kYellow    = 0xffffff00,
    kPurple    = 0xffff00ff,
    kCyan      = 0xff00ffff,
    kRed       = 0xffff0000,
    kWhite     = 0xffffffff,
    kDarkGreen = 0xff006600,
    kOrange    = 0xffffa500
    // @formatter:on
};

#ifdef USE_NVTX
inline void PUSH_RANGE(std::string const &name, Color const color) {
    auto eventAttrib = nvtxEventAttributes_t{0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = static_cast<uint32_t>(color);
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = name.c_str();
    nvtxRangePushEx(&eventAttrib);
}

inline void PUSH_RANGE(std::string const &name, uint32_t const color) {
    auto eventAttrib = nvtxEventAttributes_t{0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = color;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = name.c_str();
    nvtxRangePushEx(&eventAttrib);
}

inline void POP_RANGE(void) {
    nvtxRangePop();
}
#else // USE_NVTX
inline void PUSH_RANGE(std::string const &name, Color const color) {}
inline void PUSH_RANGE(std::string const &name, uint32_t const color) {}
inline void POP_RANGE() {}
#endif // USE_NVTX

}  // namespace nvtx
}  // namespace nie