#ifndef ULTRA_CACHE_H
#define ULTRA_CACHE_H

#include <postgres.h>
#include <lz4.h>
#include <stdint.h>

#define CACHE_TABLE_SIZE 10007  // Prime number for better distribution
#define CACHE_MAX_KEY_LENGTH 256
#define COMPRESSION_THRESHOLD 1024  // Bytes to start compression

typedef struct {
    char key[CACHE_MAX_KEY_LENGTH];
    void* value;
    size_t value_length;
    size_t original_value_length;
    bool is_compressed;
    uint8_t dist_from_start;  // Robin Hood probing distance
} CacheEntry;

typedef struct {
    CacheEntry* entries;
    size_t size;
    size_t capacity;
    float load_factor;
} RobinHoodHashTable;

#endif













































// #ifndef CACHE_H
// #define CACHE_H
// #include <postgres.h>
// #include <fmgr.h>
// #include <lz4.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>

// #define CACHE_MAX_KEY_LENGTH 256
// #define CACHE_THRESHOLD_SIZE  4096 // 4 KB instead of 1 KB
// #define CACHE_TABLE_SIZE 10007 // Prime number for hash table size

// typedef struct {
//     char key[CACHE_MAX_KEY_LENGTH]; // Key for the cache entry
//     char* value; // Pointer to the value
//     size_t value_length;
//     size_t original_value_length; // Length of the value
//     bool is_compressed; // Whether the value is compressed
// } CacheEntry;

// typedef struct {
//     CacheEntry* entries; // Array of cache entries
//     int size; // Current number of cache entries
// } UltraCache;

// // PostgreSQL function prototypes
// extern Datum ultra_cache_set(PG_FUNCTION_ARGS);
// extern Datum ultra_cache_get(PG_FUNCTION_ARGS);
// extern Datum ultra_cache_delete(PG_FUNCTION_ARGS);
// extern Datum ultra_cache_clear(PG_FUNCTION_ARGS);

// // Internal helper function prototypes
// uint64_t hash_function(const char* key);
// CacheEntry* find_cache_entry(UltraCache* cache, const char* key);
// void compress_value(const char* value, size_t length, char** compressed, size_t* compressed_length);
// void decompress_value(const char* compressed, size_t compressed_length, char** decompressed, size_t* decompressed_length,size_t original_value_length);

// #endif // CACHE_H