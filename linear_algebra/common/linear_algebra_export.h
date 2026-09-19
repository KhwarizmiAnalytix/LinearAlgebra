/*
 * KhwarizmiAnalytix LinearAlgebra library — DLL export/import (same pattern as Logging).
 */
#pragma once

#define LINALG_VISIBILITY_ENUM

#if defined(LINALG_STATIC_DEFINE)
#define LINALG_API
#define LINALG_VISIBILITY
#define LINALG_IMPORT
#define LINALG_HIDDEN

#elif defined(LINALG_SHARED_DEFINE)
#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef LINALG_BUILDING_DLL
#define LINALG_API __declspec(dllexport)
#else
#define LINALG_API __declspec(dllimport)
#endif
#define LINALG_VISIBILITY
#define LINALG_IMPORT __declspec(dllimport)
#define LINALG_HIDDEN
#elif defined(__GNUC__) && __GNUC__ >= 4
#define LINALG_API __attribute__((visibility("default")))
#define LINALG_VISIBILITY __attribute__((visibility("default")))
#define LINALG_IMPORT __attribute__((visibility("default")))
#define LINALG_HIDDEN __attribute__((visibility("hidden")))
#else
#define LINALG_API
#define LINALG_VISIBILITY
#define LINALG_IMPORT
#define LINALG_HIDDEN
#endif

#else
#define LINALG_API
#define LINALG_VISIBILITY
#define LINALG_IMPORT
#define LINALG_HIDDEN
#endif
