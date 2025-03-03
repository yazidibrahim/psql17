
#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "storage/shmem.h"
#include "utils/memutils.h"
#include "access/htup_details.h"

#include <string.h>
#include <stdint.h>
#include <time.h>

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif

// Aggressive optimization constants
#define MICRO_CACHE_SIZE 4096
#define MICRO_DATA_CHUNK_SIZE (64 * 1024)  // 64KB per chunk
#define MAX_TOTAL_DATA_SIZE (500 * 1024 * 1024)  // 500MB maximum entry size
#define MICRO_ENTRY_LIFETIME 3600           // 1-hour entry lifetime
#define MAX_CHUNKS 8192                     // Support up to 500MB with 64KB chunks

// Advanced memory management strategy
typedef struct {
    uint64_t hash;           // Compact, unique identifier
    uint32_t total_length;   // Total data length
    uint32_t expiry;         // Timestamp for cache invalidation
    uint16_t chunk_count;    // Number of chunks used
    bool is_large_entry;     // Indicates large data storage
    void *data_ptr;          // Pointer to data storage
} __attribute__((packed, aligned(64))) MicroFastCacheEntry;

// Global cache structures
static __attribute__((aligned(64))) MicroFastCacheEntry micro_cache[MICRO_CACHE_SIZE];
static volatile uint32_t micro_cache_current_time = 0;

// Function prototypes
static inline uint64_t micro_rapid_hash(const char *key, size_t len);
Datum micro_newcache_init(PG_FUNCTION_ARGS);
Datum micro_newcache_vacuum(PG_FUNCTION_ARGS);
Datum micro_newcache_cleanup(PG_FUNCTION_ARGS);
Datum micro_newcache_set(PG_FUNCTION_ARGS);
Datum micro_newcache_get(PG_FUNCTION_ARGS);

// Hyper-optimized, branchless hash function with additional mixing
static inline __attribute__((always_inline)) uint64_t 
micro_rapid_hash(const char *key, size_t len) 
{
    register uint64_t hash = 14695981039346656037ULL;
    
    // Unrolled hash computation for better instruction-level parallelism
    for (register size_t i = 0; i < len; i++) {
        hash ^= key[i];
        hash *= 1099511628211ULL;
        hash ^= hash >> 33;
        hash *= 0xff51afd7ed558ccdULL;
        hash ^= hash >> 33;
    }
    
    return hash;
}

// Highly optimized initialization function
PG_FUNCTION_INFO_V1(micro_newcache_init);
Datum micro_newcache_init(PG_FUNCTION_ARGS)
{
    // Use memset for cache initialization
    memset(micro_cache, 0, sizeof(micro_cache));
    
    // Set current time
    micro_cache_current_time = (uint32_t)time(NULL);
    
    PG_RETURN_BOOL(true);
}

// Ultra-fast vacuum with minimal branching
PG_FUNCTION_INFO_V1(micro_newcache_vacuum);
Datum micro_newcache_vacuum(PG_FUNCTION_ARGS)
{
    register uint32_t now = (uint32_t)time(NULL);
    register int removed_count = 0;
    register int i;

    for (i = 0; i < MICRO_CACHE_SIZE; i++) {
        // Branchless expiry and cleanup
        if (micro_cache[i].expiry > 0 && micro_cache[i].expiry <= now) {
            // Free dynamically allocated memory if exists
            if (micro_cache[i].is_large_entry && micro_cache[i].data_ptr) {
                pfree(micro_cache[i].data_ptr);
            }
            
            // Reset entry
            micro_cache[i].hash = 0;
            micro_cache[i].total_length = 0;
            micro_cache[i].expiry = 0;
            micro_cache[i].chunk_count = 0;
            micro_cache[i].is_large_entry = false;
            micro_cache[i].data_ptr = NULL;
            
            removed_count++;
        }
    }
    
    PG_RETURN_INT32(removed_count);
}

PG_FUNCTION_INFO_V1(micro_newcache_cleanup);
Datum micro_newcache_cleanup(PG_FUNCTION_ARGS)
{
    // Free all dynamically allocated memory for large entries
    for (int i = 0; i < MICRO_CACHE_SIZE; i++) {
        // Check if there's a large entry and free dynamically allocated memory
        if (micro_cache[i].is_large_entry && micro_cache[i].data_ptr) {
            pfree(micro_cache[i].data_ptr);
            micro_cache[i].data_ptr = NULL;  // Reset the pointer after freeing
        }
    }

    // Instead of resetting the entire cache with memset, we will reset entries individually
    for (int i = 0; i < MICRO_CACHE_SIZE; i++) {
        micro_cache[i].hash = 0;
        micro_cache[i].total_length = 0;
        micro_cache[i].expiry = 0;
        micro_cache[i].chunk_count = 0;
        micro_cache[i].is_large_entry = false;
    }

    PG_RETURN_BOOL(true);
}


// Ultra-fast set function with advanced memory management
PG_FUNCTION_INFO_V1(micro_newcache_set);
Datum micro_newcache_set(PG_FUNCTION_ARGS)
{
    // Prevent null arguments
    if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) {
        PG_RETURN_BOOL(false);
    }

    // Fetch text arguments
    text *key_arg = PG_GETARG_TEXT_P(0);
    text *value_arg = PG_GETARG_TEXT_P(1);
    
    // Get key and value pointers
    char *key = VARDATA_ANY(key_arg);
    char *value = VARDATA_ANY(value_arg);
    
    // Get lengths
    size_t key_len = VARSIZE_ANY_EXHDR(key_arg);
    size_t value_len = VARSIZE_ANY_EXHDR(value_arg);
    
    // Validate total data size
    if (value_len > MAX_TOTAL_DATA_SIZE) {
        PG_RETURN_BOOL(false);
    }
    
    // Compute hash directly
    uint64_t hash = micro_rapid_hash(key, key_len);
    uint32_t index = hash & (MICRO_CACHE_SIZE - 1);
    
    // Get cache entry
    MicroFastCacheEntry *entry = &micro_cache[index];
    
    // Clean up any existing large entry
    if (entry->is_large_entry && entry->data_ptr) {
        pfree(entry->data_ptr);
    }
    
    // Determine storage strategy
    if (value_len <= MICRO_DATA_CHUNK_SIZE) {
        // Small data: use inline storage within the entry structure
        entry->is_large_entry = false;
        entry->data_ptr = (char*)entry + sizeof(MicroFastCacheEntry);
        memcpy(entry->data_ptr, value, value_len);
    } else {
        // Large data: allocate dynamic memory
        entry->is_large_entry = true;
        entry->data_ptr = palloc(value_len);
        memcpy(entry->data_ptr, value, value_len);
    }
    
    // Update entry metadata
    entry->hash = hash;
    entry->total_length = value_len;
    entry->expiry = micro_cache_current_time + MICRO_ENTRY_LIFETIME;
    entry->chunk_count = (value_len + MICRO_DATA_CHUNK_SIZE - 1) / MICRO_DATA_CHUNK_SIZE;
    
    PG_RETURN_BOOL(true);
}

// Ultra-fast retrieval function
PG_FUNCTION_INFO_V1(micro_newcache_get);
Datum micro_newcache_get(PG_FUNCTION_ARGS)
{
    // Prevent null arguments
    if (PG_ARGISNULL(0)) {
        PG_RETURN_NULL();
    }

    // Fetch text argument
    text *key_arg = PG_GETARG_TEXT_P(0);
    
    // Get key pointer
    char *key = VARDATA_ANY(key_arg);
    size_t key_len = VARSIZE_ANY_EXHDR(key_arg);
    
    // Compute hash directly
    uint64_t hash = micro_rapid_hash(key, key_len);
    uint32_t index = hash & (MICRO_CACHE_SIZE - 1);
    
    // Get cache entry
    MicroFastCacheEntry *entry = &micro_cache[index];
    
    // Validate entry: matching hash, not expired, and has content
    if (entry->hash == hash && 
        entry->expiry > micro_cache_current_time && 
        entry->total_length > 0) {
        // Return text based on storage type
        text *result_text = cstring_to_text_with_len(
            entry->is_large_entry ? (char*)entry->data_ptr : 
                ((char*)entry + sizeof(MicroFastCacheEntry)), 
            entry->total_length
        );
        
        PG_RETURN_TEXT_P(result_text);
    }
    
    PG_RETURN_NULL();
}


