#pragma once

/*
 * Ubuntu 26.04's glibc exposes C23/GNU rsqrt declarations that conflict with
 * CUDA 13.1's math_functions.h.  Disabling _GNU_SOURCE hides those declarations,
 * but GCC 15's libstdc++ configuration was generated with GNU pthread clock
 * extensions enabled.  Include c++config first, then disable only the two
 * libstdc++ feature switches whose pthread declarations are no longer visible.
 */
#include <bits/c++config.h>

#undef _GLIBCXX_USE_PTHREAD_COND_CLOCKWAIT
#undef _GLIBCXX_USE_PTHREAD_MUTEX_CLOCKLOCK
#undef _GLIBCXX_USE_PTHREAD_RWLOCK_CLOCKLOCK
