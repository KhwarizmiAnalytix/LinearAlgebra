# Keep the bundled dependency buildable with the MSVC 19.14 baseline without
# modifying submodule checkouts. Remove this overlay once Logging/fmt support
# that compiler directly again.
function(linalg_apply_logging_msvc_compatibility)
    set(_logging_source "${LINALG_SOURCE_DIR}/ThirdParty/Logging")
    set(_compat "${LINALG_BINARY_DIR}/ThirdParty/logging-msvc-compat")

    # MSVC 19.14 predates __builtin_FILE/LINE. Its default arguments can only
    # report the declaration location; explicit file/line arguments still work.
    file(READ "${_logging_source}/include/logger/structured.h" _structured)
    string(REPLACE "__builtin_FILE()" "__FILE__" _structured "${_structured}")
    string(REPLACE "__builtin_LINE()" "__LINE__" _structured "${_structured}")
    file(MAKE_DIRECTORY "${_compat}/include/logger")
    file(WRITE "${_compat}/include/logger/structured.h" "${_structured}")

    # fmt's headers include each other by relative paths, so copy the complete
    # header set before replacing core.h. The functional enum cast in the
    # variadic descriptor trips the old MSVC parser; an explicit cast avoids it.
    file(COPY "${_logging_source}/ThirdParty/fmt/include/fmt" DESTINATION "${_compat}/fmt-include")
    file(READ "${_compat}/fmt-include/fmt/core.h" _core)
    string(REPLACE "unsigned(stored_type_constant<First, Context>::value)"
                   "static_cast<unsigned>(stored_type_constant<First, Context>::value)" _core
                   "${_core}"
    )
    # MSVC 19.14 cannot evaluate the checker holding pointers to its own
    # members. Evaluate explicit compile-string checks at runtime on this
    # compiler while retaining constexpr in the rest of fmt.
    string(REPLACE "FMT_CONSTEXPR int x = (parse_format_string(sv, checker(sv, arg_pack())), 0);"
                   "const int x = (parse_format_string(sv, checker(sv, arg_pack())), 0);" _core
                   "${_core}"
    )
    file(WRITE "${_compat}/fmt-include/fmt/core.h" "${_core}")

    target_include_directories(Logging BEFORE PRIVATE "${_compat}/fmt-include")
    target_include_directories(Logging BEFORE PUBLIC "$<BUILD_INTERFACE:${_compat}>")
endfunction()
