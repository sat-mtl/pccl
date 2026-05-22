// PCCL_API: visibility/export marker. Internal symbols are hidden;
// PCCL_API names stay exported when pccl is linked into a shared object.

#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(PCCL_BUILD_SHARED) && defined(PCCL_BUILDING)
#define PCCL_API __declspec(dllexport)
#elif defined(PCCL_BUILD_SHARED)
#define PCCL_API __declspec(dllimport)
#else
#define PCCL_API
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define PCCL_API __attribute__((visibility("default")))
#else
#define PCCL_API
#endif
