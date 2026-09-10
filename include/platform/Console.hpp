#pragma once

#if defined(_WIN32)
    /*
        windows.h defines min and max as macros, which then break every
        std::min / std::max in whatever includes this afterwards. NOMINMAX
        stops that; WIN32_LEAN_AND_MEAN drops the sockets and RPC headers
        nothing here needs
    */
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#endif

namespace platform {

/*
    useUtf8Console():
    - a Windows console defaults to an OEM code page (437 on a US install),
      so a program writing UTF-8 has its bytes rendered one at a time through
      a table that has nothing to do with unicode. The output is not wrong,
      it is being decoded wrong: the three bytes E2 80 99 of a right single
      quote come out as the three characters that 437 assigns to E2, 80 and
      99, which is why an apostrophe shows up as a Greek gamma and friends.
    - setting both code pages to 65001 makes the console read and write UTF-8,
      after which the bytes and the glyphs agree.
    - a no-op everywhere else, since every other platform is already UTF-8
    - returns whether the change took effect. It can fail when stdout is a
      pipe or a file rather than a console, which is harmless: redirected
      bytes were never being reinterpreted in the first place
*/
inline bool useUtf8Console() {
#if defined(_WIN32)
    const bool out = SetConsoleOutputCP(CP_UTF8) != 0;
    const bool in  = SetConsoleCP(CP_UTF8) != 0;
    return out && in;
#else
    return true;
#endif
}

} // !namespace platform
