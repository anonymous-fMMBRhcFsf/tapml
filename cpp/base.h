/*!
 *  Copyright (c) 2023 by Contributors
 * \file base.h
 */

#ifndef TapML_DLL
#ifdef _WIN32
#ifdef TapML_EXPORTS
#define TapML_DLL __declspec(dllexport)
#else
#define TapML_DLL __declspec(dllimport)
#endif
#else
#define TapML_DLL __attribute__((visibility("default")))
#endif
#endif
