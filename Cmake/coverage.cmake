include_guard(GLOBAL)

function(linalg_configure_coverage target)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:--coverage;-g;-O0>")
        target_link_options(${target} PUBLIC --coverage)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
            set(_debug_flags /Zi /Od)
        else()
            set(_debug_flags -g -O0)
        endif()
        target_compile_options(
            ${target}
            PRIVATE
                "$<$<COMPILE_LANGUAGE:CXX>:${_debug_flags};-fprofile-instr-generate;-fcoverage-mapping>"
        )
        target_link_options(${target} PUBLIC -fprofile-instr-generate)
    elseif(MSVC)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/Zi>")
        target_link_options(${target} PRIVATE /DEBUG:FULL /INCREMENTAL:NO)
    else()
        message(FATAL_ERROR "Coverage unsupported for ${CMAKE_CXX_COMPILER_ID}")
    endif()
endfunction()
