include_guard(GLOBAL)

function(linalg_configure_sanitizer target prefix)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(FATAL_ERROR "${prefix}: sanitizers require GCC or Clang")
    endif()
    set(_types "${${prefix}_SANITIZER_TYPE}")
    if(NOT _types)
        message(FATAL_ERROR "${prefix}_SANITIZER_TYPE must name at least one sanitizer")
    endif()
    string(REPLACE "," ";" _type_list "${_types}")
    foreach(_type IN LISTS _type_list)
        if(NOT _type MATCHES "^(address|undefined|thread|memory|leak)$")
            message(FATAL_ERROR "Unsupported sanitizer: ${_type}")
        endif()
    endforeach()
    if("address" IN_LIST _type_list)
        if("thread" IN_LIST _type_list OR "memory" IN_LIST _type_list)
            message(FATAL_ERROR "Incompatible sanitizer combination: ${_types}")
        endif()
    endif()
    target_compile_options(
        ${target}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-O1;-g;-fno-omit-frame-pointer;-fno-optimize-sibling-calls;-fsanitize=${_types}>"
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        configure_file(
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../Scripts/suppressions/sanitizer_ignore.txt.in"
            "${LINALG_BINARY_DIR}/sanitizer_ignore.txt" @ONLY
        )
        target_compile_options(
            ${target}
            PRIVATE
                "$<$<COMPILE_LANGUAGE:CXX>:-fsanitize-blacklist=${LINALG_BINARY_DIR}/sanitizer_ignore.txt>"
        )
    endif()
    target_link_options(${target} PUBLIC "-fsanitize=${_types}")
    if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND "address" IN_LIST _type_list)
        # CMake expresses the dynamic CRT with -D_DLL/--dependent-lib for the
        # GNU driver. Clang's ASan linker selection only recognizes /MD, so
        # explicitly use the DLL runtime for the library and its consumers.
        target_link_options(${target} PUBLIC -shared-libasan)
    endif()
endfunction()
