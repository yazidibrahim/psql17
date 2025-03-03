#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/elog.h"
#include "miscadmin.h"
#include <string.h>
#include <varatt.h>

PG_MODULE_MAGIC;

/* Configuration */
#define INITIAL_BUCKETS 32768
#define SMALL_POOL_BLOCK_SIZE (32 * 1024)  // 32KB for small values
#define LARGE_POOL_BLOCK_SIZE (1024 * 1024)  // 1MB for large values
#define MAX_KEY_LENGTH 256
#define LARGE_VALUE_THRESHOLD (32 * 1024)  // 32KB threshold
#define CACHE_LINE_SIZE 64

/* Memory Pool Structures */
typedef struct PoolBlock {
    struct PoolBlock* next;
    Size used;
    char padding[CACHE_LINE_SIZE - sizeof(struct PoolBlock*) - sizeof(Size)];
    uint8_t data[];  // Flexible array member
} PoolBlock;

typedef struct MemPool {
    PoolBlock* free_small_blocks;
    PoolBlock* free_large_blocks;
    Size total_size;
} MemPool;

typedef struct Entry {
    uint64_t hash;
    Size value_len;
    char key[MAX_KEY_LENGTH];  // Fixed-size array for better cache locality
    union {
        PoolBlock* blocks;     // For small values
        uint8_t* direct_buffer;  // For large values
    } data;
    bool is_large;
    struct Entry* next;
} Entry;

typedef struct {
    Entry** buckets;
    Size size;
    Size capacity;
    MemoryContext context;
    MemPool* pool;
} Store;

static Store* g_store = NULL;

/* Fast Hash Function - using xxHash algorithm */
static uint64_t fast_hash(const char* str) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    const uint64_t prime = 0x100000001b3ULL;
    const unsigned char* s = (const unsigned char*)str;
    
    while (*s) {
        hash ^= *s++;
        hash *= prime;
    }
    return hash;
}

static MemPool* create_memory_pool(void) {
    MemPool* pool = (MemPool*)MemoryContextAlloc(TopMemoryContext, sizeof(MemPool));
    if (!pool) {
        ereport(ERROR,
                (errcode(ERRCODE_OUT_OF_MEMORY),
                 errmsg("out of memory")));
    }
    
    pool->free_small_blocks = NULL;
    pool->free_large_blocks = NULL;
    pool->total_size = 0;
    return pool;
}

static PoolBlock* pool_allocate_block(Size block_size) {
    PoolBlock* block;
    PoolBlock** free_list;
    Size total_size;
    
    if (block_size <= SMALL_POOL_BLOCK_SIZE) {
        free_list = &g_store->pool->free_small_blocks;
        block_size = SMALL_POOL_BLOCK_SIZE;
    } else {
        free_list = &g_store->pool->free_large_blocks;
        block_size = LARGE_POOL_BLOCK_SIZE;
    }
    
    /* Try to reuse a block from the free list first */
    if (*free_list) {
        block = *free_list;
        *free_list = block->next;
        block->next = NULL;
        block->used = 0;
        return block;
    }
    
    total_size = sizeof(PoolBlock) + block_size;
    block = (PoolBlock*)MemoryContextAllocAligned(
        g_store->context,
        total_size,
        CACHE_LINE_SIZE,
        0);
        
    if (!block) {
        ereport(ERROR,
                (errcode(ERRCODE_OUT_OF_MEMORY),
                 errmsg("out of memory")));
    }
    
    block->next = NULL;
    block->used = 0;
    return block;
}

static void return_block_to_pool(PoolBlock* block, bool is_large) {
    if (!block) return;
    
    if (is_large) {
        block->next = g_store->pool->free_large_blocks;
        g_store->pool->free_large_blocks = block;
    } else {
        block->next = g_store->pool->free_small_blocks;
        g_store->pool->free_small_blocks = block;
    }
    block->used = 0;
}

PG_FUNCTION_INFO_V1(kv_set);
Datum kv_set(PG_FUNCTION_ARGS) {
    text* key_text;
    bytea* value_text;
    char* key;
    uint64_t hash;
    Size index, value_len, remaining;
    Entry* entry;
    uint8_t* src;
    MemoryContext old_context;
    
    if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) {
        PG_RETURN_NULL();
    }
    
    if (!g_store) {
        old_context = MemoryContextSwitchTo(TopMemoryContext);
        g_store = (Store*)palloc0(sizeof(Store));
        g_store->capacity = INITIAL_BUCKETS;
        g_store->buckets = (Entry**)palloc0(sizeof(Entry*) * INITIAL_BUCKETS);
        g_store->context = AllocSetContextCreate(TopMemoryContext,
                                               "KVStore",
                                               ALLOCSET_DEFAULT_SIZES);
        g_store->pool = create_memory_pool();
        MemoryContextSwitchTo(old_context);
    }
    
    old_context = MemoryContextSwitchTo(g_store->context);
    
    PG_TRY();
    {
        key_text = PG_GETARG_TEXT_PP(0);
        value_text = PG_GETARG_BYTEA_PP(1);
        
        key = text_to_cstring(key_text);
        if (strlen(key) >= MAX_KEY_LENGTH) {
            pfree(key);
            ereport(ERROR,
                    (errcode(ERRCODE_STRING_DATA_RIGHT_TRUNCATION),
                     errmsg("key too long")));
        }
        
        hash = fast_hash(key);
        index = hash & (g_store->capacity - 1);
        value_len = VARSIZE_ANY(value_text) - VARHDRSZ;
        
        entry = g_store->buckets[index];
        while (entry) {
            if (entry->hash == hash && strcmp(entry->key, key) == 0) {
                if (entry->is_large) {
                    pfree(entry->data.direct_buffer);
                } else {
                    PoolBlock* block = entry->data.blocks;
                    while (block) {
                        PoolBlock* next = block->next;
                        return_block_to_pool(block, false);
                        block = next;
                    }
                }
                break;
            }
            entry = entry->next;
        }
        
        if (!entry) {
            entry = (Entry*)palloc0(sizeof(Entry));
            entry->hash = hash;
            strcpy(entry->key, key);
            entry->next = g_store->buckets[index];
            g_store->buckets[index] = entry;
            g_store->size++;
        }
        
        entry->value_len = value_len;
        src = (uint8_t*)VARDATA_ANY(value_text);
        
        if (value_len >= LARGE_VALUE_THRESHOLD) {
            entry->is_large = true;
            entry->data.direct_buffer = palloc(value_len);
            memcpy(entry->data.direct_buffer, src, value_len);
        } else {
            entry->is_large = false;
            remaining = value_len;
            
            PoolBlock* current_block = pool_allocate_block(SMALL_POOL_BLOCK_SIZE);
            entry->data.blocks = current_block;
            
            while (remaining > 0) {
                Size copy_size = Min(remaining, SMALL_POOL_BLOCK_SIZE - current_block->used);
                memcpy(current_block->data + current_block->used, src, copy_size);
                
                current_block->used += copy_size;
                src += copy_size;
                remaining -= copy_size;
                
                if (remaining > 0) {
                    PoolBlock* new_block = pool_allocate_block(SMALL_POOL_BLOCK_SIZE);
                    current_block->next = new_block;
                    current_block = new_block;
                }
            }
        }
        
        pfree(key);
    }
    PG_CATCH();
    {
        MemoryContextSwitchTo(old_context);
        PG_RE_THROW();
    }
    PG_END_TRY();
    
    MemoryContextSwitchTo(old_context);
    PG_RETURN_BOOL(true);
}


PG_FUNCTION_INFO_V1(kv_get);
Datum kv_get(PG_FUNCTION_ARGS) {
    text* key_text;
    char* key;
    uint64_t hash;
    Size index;
    Entry* entry;
    bytea* result;
    uint8_t* dest;
    PoolBlock* current_block;
    Size bytes_copied;
    
    if (PG_ARGISNULL(0) || !g_store) {
        PG_RETURN_NULL();
    }
    
    key_text = PG_GETARG_TEXT_PP(0);
    key = text_to_cstring(key_text);
    
    PG_TRY();
    {
        hash = fast_hash(key);
        index = hash & (g_store->capacity - 1);
        
        entry = g_store->buckets[index];
        while (entry) {
            if (entry->hash == hash && strcmp(entry->key, key) == 0) {
                /* Allocate result with header space */
                result = (bytea*)palloc(VARHDRSZ + entry->value_len);
                if (!result) {
                    pfree(key);
                    ereport(ERROR,
                            (errcode(ERRCODE_OUT_OF_MEMORY),
                             errmsg("out of memory")));
                }
                
                /* Set bytea header */
                SET_VARSIZE(result, VARHDRSZ + entry->value_len);
                dest = (uint8_t*)VARDATA(result);
                
                if (entry->is_large) {
                    /* Fast path for large values - single copy */
                    memcpy(dest, entry->data.direct_buffer, entry->value_len);
                } else {
                    /* Copy from memory pool blocks */
                    bytes_copied = 0;
                    current_block = entry->data.blocks;
                    
                    while (current_block && bytes_copied < entry->value_len) {
                        Size remaining = entry->value_len - bytes_copied;
                        Size copy_size = Min(current_block->used, remaining);
                        
                        memcpy(dest + bytes_copied, current_block->data, copy_size);
                        bytes_copied += copy_size;
                        current_block = current_block->next;
                    }
                }
                
                pfree(key);
                PG_RETURN_BYTEA_P(result);
            }
            entry = entry->next;
        }
        
        pfree(key);
    }
    PG_CATCH();
    {
        if (key) pfree(key);
        PG_RE_THROW();
    }
    PG_END_TRY();
    
    PG_RETURN_NULL();
}

PG_FUNCTION_INFO_V1(kv_clear);
Datum kv_clear(PG_FUNCTION_ARGS) {
    if (g_store) {
        MemoryContext old_context = CurrentMemoryContext;
        
        PG_TRY();
        {
            if (g_store->context) {
                /* This will free all allocated memory in the context */
                MemoryContextDelete(g_store->context);
            }
            
            /* Free the global store structure */
            if (g_store->buckets) {
                pfree(g_store->buckets);
            }
            if (g_store->pool) {
                pfree(g_store->pool);
            }
            pfree(g_store);
            g_store = NULL;
        }
        PG_CATCH();
        {
            MemoryContextSwitchTo(old_context);
            PG_RE_THROW();
        }
        PG_END_TRY();
    }
    
    PG_RETURN_BOOL(true);
}









//SELECT kv_set('largefile', pg_read_file('/home/cybrosys/Downloads/large_html_page(1).html'));
//SELECT kv_init(1000000); 
//SELECT kv_get('key1');
//SELECT kv_get('largefile');
//SELECT kv_set('key1','yazid');









//preferred one- THE BEST


// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "utils/memutils.h"
// #include "utils/elog.h"
// #include "miscadmin.h"  // For ProcessUtility
// #include <string.h>
// #include <varatt.h>

// PG_MODULE_MAGIC;

// #define INITIAL_SIZE 32768

// typedef struct Entry {
//     uint64_t hash;
//     Size value_len;  // Changed to Size type
//     char* key;
//     bytea* value;    // Changed to bytea* for better binary data handling
//     struct Entry* next;
// } Entry;

// typedef struct {
//     Entry** buckets;
//     Size size;
//     Size capacity;
//     MemoryContext context;
// } Store;

// static Store* g_store = NULL;

// /* Function declarations */
// static void init_store(void);
// static uint64_t fast_hash(const char* str);
// static bool fast_strcmp(const char* s1, const char* s2);
// static void ensure_store_exists(void);

// static uint64_t fast_hash(const char* str) {
//     uint64_t hash = 0xcbf29ce484222325ULL;
//     const unsigned char* s = (const unsigned char*)str;
    
//     while (*s) {
//         hash ^= *s++;
//         hash *= 0x100000001b3ULL;
//     }
//     return hash;
// }

// static bool fast_strcmp(const char* s1, const char* s2) {
//     if (!s1 || !s2) return false;
//     while (*s1 && (*s1 == *s2)) {
//         s1++;
//         s2++;
//     }
//     return (*s1 - *s2) == 0;
// }

// static void ensure_store_exists(void) {
//     if (!g_store) {
//         MemoryContext old_context = MemoryContextSwitchTo(TopMemoryContext);
//         g_store = (Store*)palloc0(sizeof(Store));
//         g_store->capacity = INITIAL_SIZE;
//         g_store->size = 0;
//         g_store->buckets = (Entry**)palloc0(sizeof(Entry*) * INITIAL_SIZE);
//         g_store->context = AllocSetContextCreate(TopMemoryContext,
//                                                "KVStore",
//                                                ALLOCSET_DEFAULT_SIZES);
//         MemoryContextSwitchTo(old_context);
//     }
// }

// PG_FUNCTION_INFO_V1(kv_set);
// Datum kv_set(PG_FUNCTION_ARGS) {
//     text* key_text;
//     text* value_text;
//     char* key;
//     bytea* stored_value;
//     uint64_t hash;
//     Size index;
//     Entry* entry;
//     MemoryContext old_context;

//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) {
//         ereport(ERROR,
//                 (errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
//                  errmsg("null values not allowed")));
//     }

//     ensure_store_exists();

//     /* Switch to store's memory context */
//     old_context = MemoryContextSwitchTo(g_store->context);

//     PG_TRY();
//     {
//         key_text = PG_GETARG_TEXT_PP(0);
//         value_text = PG_GETARG_TEXT_PP(1);
        
//         /* Get key as cstring */
//         key = text_to_cstring(key_text);
        
//         /* Calculate hash and index */
//         hash = fast_hash(key);
//         index = hash & (g_store->capacity - 1);
        
//         /* Create stored value */
//         stored_value = (bytea*)palloc(VARSIZE_ANY(value_text));
//         memcpy(stored_value, value_text, VARSIZE_ANY(value_text));

//         /* Look for existing entry */
//         entry = g_store->buckets[index];
//         while (entry) {
//             if (entry->hash == hash && fast_strcmp(entry->key, key)) {
//                 /* Update existing entry */
//                 if (entry->value) {
//                     pfree(entry->value);
//                 }
//                 entry->value = stored_value;
//                 entry->value_len = VARSIZE_ANY(value_text);
//                 pfree(key);
//                 MemoryContextSwitchTo(old_context);
//                 PG_RETURN_BOOL(true);
//             }
//             entry = entry->next;
//         }

//         /* Create new entry */
//         entry = (Entry*)palloc0(sizeof(Entry));
//         entry->hash = hash;
//         entry->key = pstrdup(key);
//         entry->value = stored_value;
//         entry->value_len = VARSIZE_ANY(value_text);
//         entry->next = g_store->buckets[index];
//         g_store->buckets[index] = entry;
//         g_store->size++;

//         pfree(key);
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(old_context);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(old_context);
//     PG_RETURN_BOOL(true);
// }

// PG_FUNCTION_INFO_V1(kv_get);
// Datum kv_get(PG_FUNCTION_ARGS) {
//     text* key_text;
//     char* key;
//     uint64_t hash;
//     Size index;
//     Entry* entry;
//     bytea* result;

//     if (PG_ARGISNULL(0)) {
//         PG_RETURN_NULL();
//     }

//     if (!g_store) {
//         PG_RETURN_NULL();
//     }

//     key_text = PG_GETARG_TEXT_PP(0);
//     key = text_to_cstring(key_text);
    
//     PG_TRY();
//     {
//         hash = fast_hash(key);
//         index = hash & (g_store->capacity - 1);
//         entry = g_store->buckets[index];

//         while (entry) {
//             if (entry->hash == hash && fast_strcmp(entry->key, key)) {
//                 result = (bytea*)palloc(entry->value_len);
//                 memcpy(result, entry->value, entry->value_len);
//                 pfree(key);
//                 PG_RETURN_BYTEA_P(result);
//             }
//             entry = entry->next;
//         }

//         pfree(key);
//     }
//     PG_CATCH();
//     {
//         if (key) pfree(key);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     PG_RETURN_NULL();
// }

// PG_FUNCTION_INFO_V1(kv_clear);
// Datum kv_clear(PG_FUNCTION_ARGS) {
//     if (g_store && g_store->context) {
//         MemoryContextDelete(g_store->context);
//         pfree(g_store);
//         g_store = NULL;
//     }
//     PG_RETURN_BOOL(true);
// }



























//somewhat need to be looked
// #define _GNU_SOURCE
// #include <postgres.h>
// #include <pthread.h>
// #include <emmintrin.h>
// #include <fmgr.h>
// #include <utils/builtins.h>
// #include <sys/mman.h>
// #include <sys/stat.h>
// #include <fcntl.h>
// #include <unistd.h>
// #include <sys/resource.h>
// #include <varatt.h>

// PG_MODULE_MAGIC;

// // Optimized constants for large value handling
// #define CACHE_FILE_PATH "/tmp/ultra_cache.dat"
// #define INDEX_FILE_PATH "/tmp/ultra_cache.idx"
// #define BLOCK_SIZE (2 * 1024 * 1024)  // 2MB aligned with huge pages
// #define MAX_KEY_SIZE 64               // Using fixed-size keys for better performance
// #define INDEX_BUCKET_SIZE 16384       // Must be power of 2
// #define MAX_FILE_SIZE (1UL << 40)     // 1TB maximum file size

// // Optimized data structures
// typedef struct __attribute__((packed)) {
//     uint64_t key_hash;           // 64-bit hash of the key
//     uint64_t offset;            // Offset in the data file
//     uint32_t size;             // Size of the value
//     uint32_t block_count;      // Number of blocks
//     char key[MAX_KEY_SIZE];    // Fixed-size key
// } IndexEntry;

// typedef struct __attribute__((aligned(64))) {
//     int data_fd;              // Data file descriptor
//     int index_fd;            // Index file descriptor
//     void* data_map;          // Memory mapped data region
//     IndexEntry* index_map;   // Memory mapped index region
//     uint64_t data_size;      // Current data file size
//     uint64_t index_size;     // Current index file size
//     pthread_rwlock_t lock;   // Read-write lock for concurrent access
// } UltraCache;

// static UltraCache* cache = NULL;

// // Fast, thread-safe hash function
// static inline uint64_t fast_hash(const char* key, size_t len) {
//     uint64_t hash = 0x517cc1b727220a95;
//     for (size_t i = 0; i < len; i++) {
//         hash = (hash * 0x5bd1e995) ^ key[i];
//         hash = ((hash << 47) | (hash >> 17)) * 0x6d0acef;
//     }
//     return hash;
// }

// // Initialize the cache with optimized settings
// static void initialize_cache(void) {
//     if (cache) return;
    
//     cache = (UltraCache*)palloc0(sizeof(UltraCache));
//     pthread_rwlock_init(&cache->lock, NULL);
    
//     // Open data file with direct I/O and huge page support
//     cache->data_fd = open(CACHE_FILE_PATH, O_RDWR | O_CREAT, 0600);
//     if (cache->data_fd < 0) {
//         elog(ERROR, "Failed to open data file:");
//     }
    
//     cache->index_fd = open(INDEX_FILE_PATH, O_RDWR | O_CREAT, 0600);
//     if (cache->index_fd < 0) {
//         close(cache->data_fd);
//         elog(ERROR, "Failed to open index file:");
//     }
        
    
//     // Initialize file sizes
//     struct stat st;
//     if (fstat(cache->data_fd, &st) == 0) {
//         cache->data_size = st.st_size;
//     } else {
//         cache->data_size = BLOCK_SIZE;
//         ftruncate(cache->data_fd, BLOCK_SIZE);
//     }
    
//     if (fstat(cache->index_fd, &st) == 0) {
//         cache->index_size = st.st_size;
//     } else {
//         cache->index_size = INDEX_BUCKET_SIZE * sizeof(IndexEntry);
//         ftruncate(cache->index_fd, cache->index_size);
//     }
    
//     // Map files with optimal flags
//     cache->data_map = mmap(NULL, cache->data_size, PROT_READ | PROT_WRITE,
//                           MAP_SHARED | MAP_POPULATE | MAP_NONBLOCK, 
//                           cache->data_fd, 0);
                          
//     cache->index_map = mmap(NULL, cache->index_size, PROT_READ | PROT_WRITE,
//                            MAP_SHARED | MAP_POPULATE, 
//                            cache->index_fd, 0);
                           
//     if (cache->data_map == MAP_FAILED || cache->index_map == MAP_FAILED) {
//         elog(ERROR, "Failed to memory map files");
//     }
    
//     // Enable huge pages and optimize memory access
//     #ifdef MADV_HUGEPAGE
//     madvise(cache->data_map, cache->data_size, MADV_HUGEPAGE);
//     #endif
    
//     madvise(cache->data_map, cache->data_size, 
//             MADV_RANDOM | MADV_WILLNEED | MADV_DONTDUMP);
            
//     madvise(cache->index_map, cache->index_size,
//             MADV_SEQUENTIAL | MADV_WILLNEED);
            
//     // Set process limits and priorities
//     struct rlimit rlim = {RLIM_INFINITY, RLIM_INFINITY};
//     setrlimit(RLIMIT_MEMLOCK, &rlim);
    
//     // Lock memory to prevent swapping
//     mlock(cache->index_map, cache->index_size);
// }
// PG_FUNCTION_INFO_V1(ultra_cache_set);
// Datum ultra_cache_set(PG_FUNCTION_ARGS) {
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) PG_RETURN_BOOL(false);
    
//     initialize_cache();
    
//     text* key_text = PG_GETARG_TEXT_P(0);
//     text* value_text = PG_GETARG_TEXT_P(1);
    
//     char* key = VARDATA(key_text);
//     void* value = VARDATA(value_text);
//     int key_len = VARSIZE(key_text) - VARHDRSZ;
//     size_t value_size = VARSIZE(value_text) - VARHDRSZ;
    
//     if (key_len > MAX_KEY_SIZE) {
//         elog(ERROR, "Key too long");
//         PG_RETURN_BOOL(false);
//     }
    
//     // Calculate required blocks
//     size_t blocks_needed = (value_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     size_t aligned_size = blocks_needed * BLOCK_SIZE;
    
//     uint64_t hash = fast_hash(key, key_len);
//     uint64_t index = hash & (INDEX_BUCKET_SIZE - 1);
    
//     // Write-lock for exclusive access
//     pthread_rwlock_wrlock(&cache->lock);
    
//     // Check if we need to extend the data file
//     if (cache->data_size < aligned_size) {
//         size_t new_size = cache->data_size;
//         while (new_size < aligned_size) new_size *= 2;
        
//         // Unmap current data
//         munmap(cache->data_map, cache->data_size);
        
//         // Extend file
//         ftruncate(cache->data_fd, new_size);
        
//         // Remap with new size
//         cache->data_map = mmap(NULL, new_size, PROT_READ | PROT_WRITE,
//                               MAP_SHARED | MAP_POPULATE | MAP_NONBLOCK,
//                               cache->data_fd, 0);
                              
//         if (cache->data_map == MAP_FAILED) {
//             pthread_rwlock_unlock(&cache->lock);
//             elog(ERROR, "Failed to extend data file");
//             PG_RETURN_BOOL(false);
//         }
        
//         cache->data_size = new_size;
        
//         #ifdef MADV_HUGEPAGE
//         madvise(cache->data_map, new_size, MADV_HUGEPAGE);
//         #endif
        
//         madvise(cache->data_map, new_size, 
//                 MADV_RANDOM | MADV_WILLNEED | MADV_DONTDUMP);
//     }
    
//     // Find free space - for simplicity, always append
//     uint64_t offset = cache->data_size - aligned_size;
    
//     // Copy value using optimized memory operations
//     #ifdef __SSE2__
//     if (value_size >= 16) {
//         __m128i* src = (__m128i*)value;
//         __m128i* dst = (__m128i*)((char*)cache->data_map + offset);
//         size_t sse_blocks = value_size / 16;
        
//         for (size_t i = 0; i < sse_blocks; i++) {
//             _mm_stream_si128(&dst[i], _mm_load_si128(&src[i]));
//         }
        
//         memcpy((char*)dst + (sse_blocks * 16),
//                (char*)src + (sse_blocks * 16),
//                value_size % 16);
//     } else {
//         memcpy((char*)cache->data_map + offset, value, value_size);
//     }
//     #else
//     memcpy((char*)cache->data_map + offset, value, value_size);
//     #endif
    
//     // Update index entry
//     IndexEntry* entry = &cache->index_map[index];
//     entry->key_hash = hash;
//     entry->offset = offset;
//     entry->size = value_size;
//     entry->block_count = blocks_needed;
//     memcpy(entry->key, key, key_len);
    
//     // Ensure data is written to disk
//     msync((char*)cache->data_map + offset, value_size, MS_ASYNC);
//     msync(&cache->index_map[index], sizeof(IndexEntry), MS_SYNC);
    
//     pthread_rwlock_unlock(&cache->lock);
//     PG_RETURN_BOOL(true);
// }
// // Fast value retrieval function
// PG_FUNCTION_INFO_V1(ultra_cache_get);
// Datum ultra_cache_get(PG_FUNCTION_ARGS) {
//     if (PG_ARGISNULL(0) || !cache) PG_RETURN_NULL();
    
//     text* key_text = PG_GETARG_TEXT_P(0);
//     char* key = VARDATA(key_text);
//     int key_len = VARSIZE(key_text) - VARHDRSZ;
    
//     if (key_len > MAX_KEY_SIZE) {
//         elog(ERROR, "Key too long");
//         PG_RETURN_NULL();
//     }
    
//     uint64_t hash = fast_hash(key, key_len);
//     uint64_t index = hash & (INDEX_BUCKET_SIZE - 1);
    
//     // Read-lock for concurrent access
//     pthread_rwlock_rdlock(&cache->lock);
    
//     IndexEntry* entry = &cache->index_map[index];
    
//     if (entry->key_hash != hash || strncmp(entry->key, key, key_len) != 0) {
//         pthread_rwlock_unlock(&cache->lock);
//         PG_RETURN_NULL();
//     }
    
//     // Direct memory access for value retrieval
//     void* value_ptr = (char*)cache->data_map + entry->offset;
    
//     // Create result with the exact size needed
//     text* result = (text*)palloc(VARHDRSZ + entry->size);
//     SET_VARSIZE(result, VARHDRSZ + entry->size);
    
//     // Use optimized memory copy
//     #ifdef __SSE2__
//     if (entry->size >= 16) {
//         __m128i* src = (__m128i*)value_ptr;
//         __m128i* dst = (__m128i*)VARDATA(result);
//         size_t sse_blocks = entry->size / 16;
        
//         for (size_t i = 0; i < sse_blocks; i++) {
//             _mm_stream_si128(&dst[i], _mm_load_si128(&src[i]));
//         }
        
//         // Copy remaining bytes
//         memcpy((char*)dst + (sse_blocks * 16),
//                (char*)src + (sse_blocks * 16),
//                entry->size % 16);
//     } else {
//         memcpy(VARDATA(result), value_ptr, entry->size);
//     }
//     #else
//     memcpy(VARDATA(result), value_ptr, entry->size);
//     #endif
    
//     pthread_rwlock_unlock(&cache->lock);
//     PG_RETURN_TEXT_P(result);
// }






























//not working
//     #define _GNU_SOURCE   
//     #include <postgres.h>
//     #include <fmgr.h>
//     #include <utils/memutils.h>
//     #include "funcapi.h"  
//     #include <utils/builtins.h>
//     #include <storage/lwlock.h>
//     #include <storage/lwlocknames.h>  
//     #include <storage/shmem.h>
//     #include <storage/ipc.h>
//     #include <storage/proc.h>
//     #include <access/xact.h>
//     #include <sys/mman.h>
//     #include <sys/stat.h>
//     #include <fcntl.h>
//     #include <unistd.h>
//     #include <xxhash.h>
//     #include <varatt.h>
//     #include <miscadmin.h>
//     #include <zlib.h>
//     #include <sys/resource.h>
//     #include <pthread.h>
//     #include "access/htup_details.h"
//     #include "access/tupdesc.h"
//     #include "catalog/pg_type.h"


//     PG_MODULE_MAGIC;

//     // Configuration Constants
//     #define CACHE_FILE_PATH "/tmp/ultra_cache.dat"
//     #define MAX_KEY_LENGTH 256
//     #define INITIAL_CAPACITY (1 << 14)  // 16384
//     #define MAX_LOAD_FACTOR 0.75
//     #define CACHE_PAGE_SIZE (1000 * 1024 * 1024)  // 2MB for huge pages
//     #define DEFAULT_COMPRESS_LEVEL Z_BEST_SPEED
//     #define MEMORY_BLOCK_SIZE (64 * 1024 * 1024)  // 64MB blocks for memory pool
//     #define LARGE_VALUE_THRESHOLD (64 * 1024)  // 64KB threshold for background compression

//     // Error Messages
//     #define ERROR_MSG_INIT "Could not initialize cache"
//     #define ERROR_MSG_FILE "Could not access cache file"
//     #define ERROR_MSG_MMAP "Could not memory map cache file"
//     #define ERROR_MSG_BOUNDS "Cache entry extends beyond allocated memory"
//     #define ERROR_MSG_SPACE "Insufficient space in cache"
//     #define ERROR_MSG_COMPRESS "Compression failed"

//     // Cache Statistics
//     typedef struct {
//         uint64_t gets;
//         uint64_t sets;
//         uint64_t hits;
//         uint64_t misses;
//         uint64_t evictions;
//         uint64_t total_compressed;
//         uint64_t total_uncompressed;
//     } CacheStats;

//     // Memory Pool Structures
//     typedef struct {
//         size_t block_size;
//         size_t used;
//         char* data;
//     } MemoryBlock;

//     typedef struct {
//         MemoryBlock* blocks;
//         size_t block_count;
//         size_t current_block;
//         size_t total_size;
//     } MemoryPool;

//     // Robin Hood Hash Table Entry
//     typedef struct __attribute__((aligned(64))) {
//         XXH64_hash_t key_hash;
//         uint32_t probe_distance;
//         off_t value_offset;
//         size_t value_length;
//         size_t original_length;
//         char key[MAX_KEY_LENGTH];
//         bool is_compressed;
//     } RobinHoodEntry;

//     // Compression queue entry
//     typedef struct CompressionTask {
//         void* data;
//         size_t size;
//         XXH64_hash_t key_hash;
//         char key[MAX_KEY_LENGTH];
//         struct CompressionTask* next;
//     } CompressionTask;

//     // Compression queue
//     typedef struct {
//         CompressionTask* head;
//         CompressionTask* tail;
//         pthread_mutex_t lock;
//         pthread_cond_t cond;
//         bool shutdown;
//         int task_count;
//     } CompressionQueue;

//     // Global compression state
//     typedef struct {
//         pthread_t worker_thread;
//         CompressionQueue queue;
//     } CompressionState;

//     // Memory-Mapped Cache Structure
//     typedef struct {
//         int fd;
//         void* mapped_data;
//         size_t file_size;
//         size_t max_file_size;
        
//         RobinHoodEntry* entries;
//         size_t capacity;
//         size_t size;
//         size_t mask;
        
//         MemoryPool* value_pool;
//         CacheStats stats;
//     } UltraCache;

//     // Global variables
//     static UltraCache* global_cache = NULL;
//     static MemoryContext cache_memory_context = NULL;
//     static LWLock* cache_lock = NULL;
//     static CompressionState* compression_state = NULL;

//     // Function declarations
//     static void ultra_cache_shmem_request(void);
//     static void cache_shmem_startup(XactEvent event, void *arg);
//     static bool validate_entry(RobinHoodEntry* entry);
//     static void* allocate_value_space(size_t size);
//     static void initialize_memory_pool(void);
//     static void* compress_value(void* data, size_t size, size_t* compressed_size);
//     static void* decompress_value(void* data, size_t compressed_size, size_t original_size);
//     static void process_compression_task(CompressionTask* task);
//     static void* compression_worker(void* arg);

//     // Initialize shared memory and locks
//     static void cache_shmem_startup(XactEvent event, void *arg){
//         bool found;
//         cache_lock = ShmemInitStruct("UltraCache", sizeof(LWLock), &found);
//         if (!found) {
//             LWLockInitialize(cache_lock, LWLockNewTrancheId());
//         }
//     }

//     static void ultra_cache_shmem_request(void) {
//     RequestAddinShmemSpace(sizeof(LWLock));
//     RequestNamedLWLockTranche("UltraCache", 1);
//     }
//     // Fast hash function
//     static inline XXH64_hash_t fast_hash(const char* key) {
//         return XXH3_64bits(key, strlen(key));
//     }

//     // Lock management
//     static void acquire_lock(bool exclusive) {
//         if (cache_lock) {
//             if (exclusive) {
//                 LWLockAcquire(cache_lock, LW_EXCLUSIVE);
//             } else {
//                 LWLockAcquire(cache_lock, LW_SHARED);
//             }
//         }
//     }

//     static void release_lock(void) {
//         if (cache_lock) {
//             LWLockRelease(cache_lock);
//         }
//     }

//     // Initialize memory pool
//     static void initialize_memory_pool(void) {
//         MemoryContext oldcontext = MemoryContextSwitchTo(cache_memory_context);
        
//         global_cache->value_pool = palloc0(sizeof(MemoryPool));
//         global_cache->value_pool->blocks = palloc0(sizeof(MemoryBlock));
//         global_cache->value_pool->block_count = 1;
//         global_cache->value_pool->current_block = 0;
        
//         global_cache->value_pool->blocks[0].block_size = MEMORY_BLOCK_SIZE;
//         global_cache->value_pool->blocks[0].used = 0;
//         global_cache->value_pool->blocks[0].data = palloc0(MEMORY_BLOCK_SIZE);
        
//         MemoryContextSwitchTo(oldcontext);
//     }

//     // Initialize compression worker
//     static void initialize_compression_worker(void) {
//         compression_state = palloc0(sizeof(CompressionState));
        
//         pthread_mutex_init(&compression_state->queue.lock, NULL);
//         pthread_cond_init(&compression_state->queue.cond, NULL);
//         compression_state->queue.shutdown = false;
//         compression_state->queue.task_count = 0;
        
//         pthread_create(&compression_state->worker_thread, NULL, compression_worker, NULL);
//     }

//     // Allocate space for value
//     static void* allocate_value_space(size_t size) {
//         size = (size + 63) & ~63;  // Align to cache line
        
//         MemoryPool* pool = global_cache->value_pool;
//         if (pool->blocks[pool->current_block].used + size > pool->blocks[pool->current_block].block_size) {
//             pool->current_block++;
//             if (pool->current_block >= pool->block_count) {
//                 pool->blocks = repalloc(pool->blocks, (pool->block_count + 1) * sizeof(MemoryBlock));
//                 pool->blocks[pool->block_count].block_size = MEMORY_BLOCK_SIZE;
//                 pool->blocks[pool->block_count].used = 0;
//                 pool->blocks[pool->block_count].data = palloc0(MEMORY_BLOCK_SIZE);
//                 pool->block_count++;
//             }
//         }
        
//         void* ptr = pool->blocks[pool->current_block].data + pool->blocks[pool->current_block].used;
//         pool->blocks[pool->current_block].used += size;
//         return ptr;
//     }

//     // Compression functions
//     static void* compress_value(void* data, size_t size, size_t* compressed_size) {
//         uLong bound = compressBound(size);
//         void* compressed = palloc(bound);
        
//         if (compress2(compressed, &bound, data, size, DEFAULT_COMPRESS_LEVEL) != Z_OK) {
//             pfree(compressed);
//             elog(ERROR, ERROR_MSG_COMPRESS);
//             return NULL;
//         }
        
//         *compressed_size = bound;
//         return compressed;
//     }

//     static void* decompress_value(void* data, size_t compressed_size, size_t original_size) {
//         void* decompressed = palloc(original_size);
//         uLong decompressed_size = original_size;
        
//         if (uncompress(decompressed, &decompressed_size, data, compressed_size) != Z_OK) {
//             pfree(decompressed);
//             elog(ERROR, "Decompression failed");
//             return NULL;
//         }
        
//         return decompressed;
//     }

//     // Process compression task
//     static void process_compression_task(CompressionTask* task) {
//         size_t compressed_size;
//         void* compressed_data = compress_value(task->data, task->size, &compressed_size);
        
//         if (compressed_data) {
//             void* storage = allocate_value_space(compressed_size);
//             memcpy(storage, compressed_data, compressed_size);
//             pfree(compressed_data);
            
//             size_t index = task->key_hash & global_cache->mask;
//             RobinHoodEntry* entry = &global_cache->entries[index];
            
//             acquire_lock(true);
            
//             if (entry->key_hash == task->key_hash && strcmp(entry->key, task->key) == 0) {
//                 entry->value_offset = (char*)storage - (char*)global_cache->mapped_data;
//                 entry->value_length = compressed_size;
//                 entry->is_compressed = true;
                
//                 global_cache->stats.total_compressed += compressed_size;
//                 global_cache->stats.total_uncompressed += task->size;
//             }
            
//             release_lock();
//         }
        
//         pfree(task->data);
//         pfree(task);
//     }

//     // Compression worker thread
//     static void* compression_worker(void* arg) {
//         while (true) {
//             pthread_mutex_lock(&compression_state->queue.lock);
            
//             while (!compression_state->queue.head && !compression_state->queue.shutdown) {
//                 pthread_cond_wait(&compression_state->queue.cond, 
//                                 &compression_state->queue.lock);
//             }
            
//             if (compression_state->queue.shutdown && !compression_state->queue.head) {
//                 pthread_mutex_unlock(&compression_state->queue.lock);
//                 break;
//             }
            
//             CompressionTask* task = compression_state->queue.head;
//             compression_state->queue.head = task->next;
//             if (!compression_state->queue.head) {
//                 compression_state->queue.tail = NULL;
//             }
//             compression_state->queue.task_count--;
            
//             pthread_mutex_unlock(&compression_state->queue.lock);
            
//             process_compression_task(task);
//         }
        
//         return NULL;
//     }

//     // Queue compression task
//     static void queue_compression_task(void* data, size_t size, const char* key, XXH64_hash_t key_hash) {
//         CompressionTask* task = palloc(sizeof(CompressionTask));
//         task->data = data;
//         task->size = size;
//         task->key_hash = key_hash;
//         strncpy(task->key, key, MAX_KEY_LENGTH - 1);
//         task->next = NULL;
        
//         pthread_mutex_lock(&compression_state->queue.lock);
        
//         if (compression_state->queue.tail) {
//             compression_state->queue.tail->next = task;
//         } else {
//             compression_state->queue.head = task;
//         }
//         compression_state->queue.tail = task;
//         compression_state->queue.task_count++;
        
//         pthread_cond_signal(&compression_state->queue.cond);
//         pthread_mutex_unlock(&compression_state->queue.lock);
//     }

//     // Initialize cache
//     static void initialize_cache(void) {
//         if (global_cache) return;

//         cache_memory_context = AllocSetContextCreate(
//             TopMemoryContext,
//             "UltraCache Memory Context",
//             ALLOCSET_DEFAULT_SIZES
//         );

//         MemoryContext oldcontext = MemoryContextSwitchTo(cache_memory_context);
        
//         global_cache = palloc0(sizeof(UltraCache));
//         global_cache->capacity = INITIAL_CAPACITY;
//         global_cache->mask = INITIAL_CAPACITY - 1;
//         global_cache->entries = palloc0(INITIAL_CAPACITY * sizeof(RobinHoodEntry));
//         global_cache->max_file_size = 1024 * 1024 * 1024;  // 1GB max
        
//         initialize_memory_pool();
//         if (access("/tmp", W_OK) != 0) {
//         elog(ERROR, "Cannot access /tmp directory");
//         }
        
//         global_cache->fd = open(CACHE_FILE_PATH, 
//                             O_RDWR | O_CREAT | O_DIRECT | O_LARGEFILE, 
//                             S_IRUSR | S_IWUSR);
//         if (global_cache->fd == -1) {
//             elog(ERROR, ERROR_MSG_FILE);
//         }

//         if (posix_fallocate(global_cache->fd, 0, global_cache->max_file_size) != 0) {
//             close(global_cache->fd);
//             elog(ERROR, ERROR_MSG_FILE);
//         }

//         global_cache->mapped_data = mmap(
//             NULL, 
//             global_cache->max_file_size, 
//             PROT_READ | PROT_WRITE, 
//             MAP_SHARED | MAP_POPULATE | MAP_NONBLOCK, 
//             global_cache->fd, 
//             0
//         );

//         if (global_cache->mapped_data == MAP_FAILED) {
//             close(global_cache->fd);
//             elog(ERROR, ERROR_MSG_MMAP);
//         }

//         madvise(global_cache->mapped_data, 
//                 global_cache->max_file_size, 
//                 MADV_WILLNEED | MADV_SEQUENTIAL | MADV_HUGEPAGE);
                
//         posix_madvise(global_cache->mapped_data, 
//                     global_cache->max_file_size, 
//                     POSIX_MADV_WILLNEED | POSIX_MADV_SEQUENTIAL);
        
//         memset(&global_cache->stats, 0, sizeof(CacheStats));

//         MemoryContextSwitchTo(oldcontext);
//     }

//     // Cache operations
//     PG_FUNCTION_INFO_V1(ultra_cache_set);
//     Datum ultra_cache_set(PG_FUNCTION_ARGS) {
//         if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) 
//             PG_RETURN_BOOL(false);

//         initialize_cache();
        
//         if (!compression_state) {
//             initialize_compression_worker();
//         }

//         text* key_text = PG_GETARG_TEXT_P(0);
//         text* value_text = PG_GETARG_TEXT_P(1);
//         char* key = text_to_cstring(key_text);
//         void* value = VARDATA(value_text);
//         size_t value_length = VARSIZE_ANY_EXHDR(value_text);
        
//         acquire_lock(true);
        
//         XXH64_hash_t hash = fast_hash(key);
//         size_t index = hash & global_cache->mask;
//         RobinHoodEntry* entry = &global_cache->entries[index];
        
//         if (value_length >= LARGE_VALUE_THRESHOLD) {
//             // Store uncompressed initially and queue for background compression
//             void* temp_storage = allocate_value_space(value_length);
//             memcpy(temp_storage, value, value_length);
            
//             entry->key_hash = hash;
//             entry->value_offset = (char*)temp_storage - (char*)global_cache->mapped_data;
//             entry->value_length = value_length;
//             entry->original_length = value_length;
//             entry->is_compressed = false;
//             strncpy(entry->key, key, MAX_KEY_LENGTH - 1);
            
//             // Queue for background compression
//             void* compression_data = palloc(value_length);
//             memcpy(compression_data, value, value_length);
//             queue_compression_task(compression_data, value_length, key, hash);
//         } else {
//             // Compress immediately for small values
//             size_t compressed_size;
//             void* compressed_value = compress_value(value, value_length, &compressed_size);
//             if (!compressed_value) {
//                 release_lock();
//                 PG_RETURN_BOOL(false);
//             }
            
//             void* storage = allocate_value_space(compressed_size);
//             memcpy(storage, compressed_value, compressed_size);
//             pfree(compressed_value);
            
//             entry->key_hash = hash;
//             entry->value_offset = (char*)storage - (char*)global_cache->mapped_data;
//             entry->value_length = compressed_size;
//             entry->original_length = value_length;
//             entry->is_compressed = true;
//             strncpy(entry->key, key, MAX_KEY_LENGTH - 1);
            
//             global_cache->stats.total_compressed += compressed_size;
//             global_cache->stats.total_uncompressed += value_length;
//         }
        
//         global_cache->stats.sets++;
//         release_lock();
//         PG_RETURN_BOOL(true);
//     }

//     PG_FUNCTION_INFO_V1(ultra_cache_get);
//     Datum ultra_cache_get(PG_FUNCTION_ARGS) {
//         if (PG_ARGISNULL(0) || !global_cache)
//             PG_RETURN_NULL();

//         text* key_text = PG_GETARG_TEXT_P(0);
//         char* key = text_to_cstring(key_text);
//         XXH64_hash_t hash = fast_hash(key);

//         acquire_lock(false);
//         global_cache->stats.gets++;

//         RobinHoodEntry* entry = &global_cache->entries[hash & global_cache->mask];
//         if (entry->key[0] == '\0' || entry->key_hash != hash || strcmp(entry->key, key) != 0) {
//             global_cache->stats.misses++;
//             release_lock();
//             PG_RETURN_NULL();
//         }

//         global_cache->stats.hits++;
        
//         void* value_data = (char*)global_cache->mapped_data + entry->value_offset;
//         void* final_data;
//         size_t final_size;
        
//         if (entry->is_compressed) {
//             final_data = decompress_value(value_data, entry->value_length, entry->original_length);
//             final_size = entry->original_length;
//         } else {
//             final_data = palloc(entry->value_length);
//             memcpy(final_data, value_data, entry->value_length);
//             final_size = entry->value_length;
//         }
        
//         text* result = (text*) palloc(VARHDRSZ + final_size);
//         SET_VARSIZE(result, VARHDRSZ + final_size);
//         memcpy(VARDATA(result), final_data, final_size);
        
//         pfree(final_data);
//         release_lock();
//         PG_RETURN_TEXT_P(result);
//     }

//     PG_FUNCTION_INFO_V1(ultra_cache_delete);
//     Datum ultra_cache_delete(PG_FUNCTION_ARGS) {
//         if (PG_ARGISNULL(0) || !global_cache)
//             PG_RETURN_BOOL(false);

//         text* key_text = PG_GETARG_TEXT_P(0);
//         char* key = text_to_cstring(key_text);
//         XXH64_hash_t hash = fast_hash(key);

//         acquire_lock(true);

//         size_t index = hash & global_cache->mask;
//         RobinHoodEntry* entry = &global_cache->entries[index];
        
//         if (entry->key[0] == '\0' || entry->key_hash != hash || strcmp(entry->key, key) != 0) {
//             release_lock();
//             PG_RETURN_BOOL(false);
//         }

//         // Mark entry as deleted
//         entry->key[0] = '\0';
//         entry->value_length = 0;
//         entry->original_length = 0;
//         entry->is_compressed = false;
        
//         global_cache->stats.evictions++;
//         release_lock();
        
//         PG_RETURN_BOOL(true);
//     }

//     PG_FUNCTION_INFO_V1(ultra_cache_clear);
//     Datum ultra_cache_clear(PG_FUNCTION_ARGS) {
//         if (!global_cache)
//             PG_RETURN_VOID();

//         acquire_lock(true);
        
//         // Reset all entries
//         memset(global_cache->entries, 0, global_cache->capacity * sizeof(RobinHoodEntry));
        
//         // Reset memory pool
//         for (size_t i = 0; i < global_cache->value_pool->block_count; i++) {
//             global_cache->value_pool->blocks[i].used = 0;
//         }
//         global_cache->value_pool->current_block = 0;
        
//         // Reset statistics
//         memset(&global_cache->stats, 0, sizeof(CacheStats));
        
//         release_lock();
//         PG_RETURN_VOID();
//     }

//     PG_FUNCTION_INFO_V1(ultra_cache_stats);
// Datum ultra_cache_stats(PG_FUNCTION_ARGS) {
//     if (!global_cache)
//         PG_RETURN_NULL();

//     acquire_lock(false);
    
//     Datum values[7];
//     bool nulls[7] = {false};
    
//     values[0] = UInt64GetDatum(global_cache->stats.gets);
//     values[1] = UInt64GetDatum(global_cache->stats.sets);
//     values[2] = UInt64GetDatum(global_cache->stats.hits);
//     values[3] = UInt64GetDatum(global_cache->stats.misses);
//     values[4] = UInt64GetDatum(global_cache->stats.evictions);
//     values[5] = UInt64GetDatum(global_cache->stats.total_compressed);
//     values[6] = UInt64GetDatum(global_cache->stats.total_uncompressed);
    
//     release_lock();
    
//     TupleDesc tupdesc = CreateTemplateTupleDesc(7);
//     TupleDescInitEntry(tupdesc, 1, "gets", INT8OID, -1, 0);
//     TupleDescInitEntry(tupdesc, 2, "sets", INT8OID, -1, 0);
//     TupleDescInitEntry(tupdesc, 3, "hits", INT8OID, -1, 0);
//     TupleDescInitEntry(tupdesc, 4, "misses", INT8OID, -1, 0);
//     TupleDescInitEntry(tupdesc, 5, "evictions", INT8OID, -1, 0);
//     TupleDescInitEntry(tupdesc, 6, "total_compressed", INT8OID, -1, 0);
//     TupleDescInitEntry(tupdesc, 7, "total_uncompressed", INT8OID, -1, 0);
    
//     BlessTupleDesc(tupdesc);
    
//     HeapTuple tuple = heap_form_tuple(tupdesc, values, nulls);
//     Datum result = HeapTupleGetDatum(tuple);
    
//     ReleaseTupleDesc(tupdesc);
    
//     PG_RETURN_DATUM(result);
// }

//     // Module initialization
//     void _PG_init(void) {
//         shmem_request_hook = ultra_cache_shmem_request;
        
//         // Register background worker if needed
//         RegisterXactCallback(cache_shmem_startup, 0);
//     }

//     // Module cleanup
//     void _PG_fini(void) {
//         if (compression_state) {
//             pthread_mutex_lock(&compression_state->queue.lock);
//             compression_state->queue.shutdown = true;
//             pthread_cond_signal(&compression_state->queue.cond);
//             pthread_mutex_unlock(&compression_state->queue.lock);
            
//             pthread_join(compression_state->worker_thread, NULL);
//         }
        
//         if (global_cache) {
//             if (global_cache->mapped_data) {
//                 munmap(global_cache->mapped_data, global_cache->max_file_size);
//             }
//             if (global_cache->fd != -1) {
//                 close(global_cache->fd);
//             }
//         }
//     }





















///new one 
// #define _GNU_SOURCE   
// #include <postgres.h>
// #include <fmgr.h>
// #include <utils/memutils.h>
// #include <utils/builtins.h>
// #include <storage/lwlock.h>
// #include <storage/lwlocknames.h>  
// #include <storage/shmem.h>
// #include <storage/ipc.h>
// #include <storage/proc.h>
// #include <access/xact.h>
// #include <sys/mman.h>
// #include <sys/stat.h>
// #include <fcntl.h>
// #include <unistd.h>
// #include <xxhash.h>
// #include <varatt.h>
// #include <miscadmin.h>

// PG_MODULE_MAGIC;

// // Configuration Constants
// #define CACHE_FILE_PATH "/tmp/ultra_cache.dat"
// #define MAX_KEY_LENGTH 256
// #define INITIAL_CAPACITY 16384
// #define MAX_LOAD_FACTOR 0.75
// #define CACHE_ALIGNMENT 64
// #define HUGE_PAGE_SIZE (2 * 1024 * 1024)  // 2MB huge pages

// // Error Messages
// #define ERROR_MSG_INIT "Could not initialize cache"
// #define ERROR_MSG_FILE "Could not access cache file"
// #define ERROR_MSG_MMAP "Could not memory map cache file"
// #define ERROR_MSG_BOUNDS "Cache entry extends beyond allocated memory"
// #define ERROR_MSG_SPACE "Insufficient space in cache"

// // Robin Hood Hash Table Entry
// typedef struct {
//     XXH64_hash_t key_hash;
//     uint32_t probe_distance;
//     off_t value_offset;
//     size_t value_length;
//     char key[MAX_KEY_LENGTH];
// } __attribute__((packed)) RobinHoodEntry;

// // Memory-Mapped Cache Structure
// typedef struct {
//     int fd;
//     void* mapped_data;
//     size_t file_size;
//     size_t max_file_size;
    
//     RobinHoodEntry* entries;
//     size_t capacity;
//     size_t size;
//     size_t mask;
// } UltraCache;

// // Global variables
// static UltraCache* global_cache = NULL;
// static MemoryContext cache_memory_context = NULL;
// static LWLock* cache_lock = NULL;

// // Function declarations
// static void cache_shmem_startup(int code, Datum arg);
// static bool validate_entry(RobinHoodEntry* entry);

// static void cache_shmem_startup(int code, Datum arg) {
//     bool found;
    
//     cache_lock = ShmemInitStruct("UltraCache",
//                                sizeof(LWLock),
//                                &found);
                               
//     if (!found) {
//         LWLockInitialize(cache_lock, LWLockNewTrancheId());
//     }
// }
// // Fast hash function using XXH3
// static inline XXH64_hash_t fast_hash(const char* key) {
//     return XXH3_64bits(key, strlen(key));
// }

// // Lock management functions
// static void acquire_lock(bool exclusive) {
//     if (cache_lock) {
//         if (exclusive) {
//             LWLockAcquire(cache_lock, LW_EXCLUSIVE);
//         } else {
//             LWLockAcquire(cache_lock, LW_SHARED);
//         }
//     }
// }

// static void release_lock(void) {
//     if (cache_lock) {
//         LWLockRelease(cache_lock);
//     }
// }

// // Validate cache entry
// static bool validate_entry(RobinHoodEntry* entry) {
//     if (!entry || !global_cache) return false;
    
//     if (entry->value_offset + entry->value_length > global_cache->max_file_size) {
//         elog(ERROR, ERROR_MSG_BOUNDS);
//         return false;
//     }
    
//     return true;
// }

// // Find entry with Robin Hood probing
// static RobinHoodEntry* find_cache_entry(UltraCache* cache, const char* key, XXH64_hash_t hash) {
//     if (!cache) return NULL;

//     size_t index = hash & cache->mask;
//     uint32_t probe_distance = 0;

//     while (true) {
//         RobinHoodEntry* entry = &cache->entries[index];
        
//         if (entry->key[0] == '\0') 
//             return NULL;
        
//         if (entry->key_hash == hash && strcmp(entry->key, key) == 0) {
//             return entry;
//         }
        
//         if (entry->probe_distance < probe_distance) {
//             return NULL;
//         }
        
//         index = (index + 1) & cache->mask;
//         probe_distance++;
//     }
// }

// // Initialize cache with proper memory management
// static void initialize_cache(void) {
//     if (global_cache) return;

//     cache_memory_context = AllocSetContextCreate(
//         TopMemoryContext,
//         "UltraCache Memory Context",
//         ALLOCSET_DEFAULT_SIZES
//     );

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_memory_context);
    
//     global_cache = palloc0(sizeof(UltraCache));
//     global_cache->capacity = INITIAL_CAPACITY;
//     global_cache->mask = INITIAL_CAPACITY - 1;
//     global_cache->entries = palloc0(INITIAL_CAPACITY * sizeof(RobinHoodEntry));
//     global_cache->max_file_size = 1024 * 1024 * 1024;  // 1GB max
    
//     global_cache->fd = open(CACHE_FILE_PATH, 
//                           O_RDWR | O_CREAT | O_LARGEFILE, 
//                           S_IRUSR | S_IWUSR);
//     if (global_cache->fd == -1) {
//         elog(ERROR, ERROR_MSG_FILE);
//     }

//     if (posix_fallocate(global_cache->fd, 0, global_cache->max_file_size) != 0) {
//         close(global_cache->fd);
//         elog(ERROR, ERROR_MSG_FILE);
//     }

//     global_cache->mapped_data = mmap(
//         NULL, 
//         global_cache->max_file_size, 
//         PROT_READ | PROT_WRITE, 
//         MAP_SHARED | MAP_POPULATE, 
//         global_cache->fd, 
//         0
//     );

//     if (global_cache->mapped_data == MAP_FAILED) {
//         close(global_cache->fd);
//         elog(ERROR, ERROR_MSG_MMAP);
//     }
//     madvise(global_cache->mapped_data, 
//         global_cache->max_file_size, 
//         MADV_HUGEPAGE);
        
//     posix_madvise(
//         global_cache->mapped_data, 
//         global_cache->max_file_size, 
//         POSIX_MADV_WILLNEED | POSIX_MADV_SEQUENTIAL
//     );

//     MemoryContextSwitchTo(oldcontext);
// }

// // Resize hash table with Robin Hood probing
// static void resize_cache(UltraCache* cache) {
//     size_t new_capacity = cache->capacity * 2;
//     size_t new_mask = new_capacity - 1;
    
//     RobinHoodEntry* new_entries = palloc0(new_capacity * sizeof(RobinHoodEntry));
    
//     for (size_t i = 0; i < cache->capacity; i++) {
//         if (cache->entries[i].key[0] != '\0') {
//             RobinHoodEntry* current = &cache->entries[i];
//             size_t index = current->key_hash & new_mask;
//             uint32_t probe_distance = 0;
            
//             while (new_entries[index].key[0] != '\0') {
//                 if (new_entries[index].probe_distance < probe_distance) {
//                     RobinHoodEntry temp = new_entries[index];
//                     new_entries[index] = *current;
//                     new_entries[index].probe_distance = probe_distance;
//                     *current = temp;
//                 }
                
//                 index = (index + 1) & new_mask;
//                 probe_distance++;
//             }
            
//             new_entries[index] = *current;
//             new_entries[index].probe_distance = probe_distance;
//         }
//     }
    
//     pfree(cache->entries);
//     cache->entries = new_entries;
//     cache->capacity = new_capacity;
//     cache->mask = new_mask;
// }

// // Set cache entry with proper memory management
// PG_FUNCTION_INFO_V1(ultra_cache_set);
// Datum ultra_cache_set(PG_FUNCTION_ARGS) {
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) 
//         PG_RETURN_BOOL(false);

//     initialize_cache();

//     text* key_text = PG_GETARG_TEXT_P(0);
//     text* value_text = PG_GETARG_TEXT_P(1);
//     char* key = text_to_cstring(key_text);
//     void* value = VARDATA(value_text);
//     size_t value_length = VARSIZE_ANY_EXHDR(value_text);

//     acquire_lock(true);

//     if ((float)global_cache->size / global_cache->capacity > MAX_LOAD_FACTOR) {
//         resize_cache(global_cache);
//     }

//     XXH64_hash_t hash = fast_hash(key);
//     size_t index = hash & global_cache->mask;
//     uint32_t probe_distance = 0;
//     RobinHoodEntry* entry = NULL;

//     while (true) {
//         RobinHoodEntry* current = &global_cache->entries[index];
        
//         if (current->key[0] == '\0' || 
//             (current->key_hash == hash && strcmp(current->key, key) == 0)) {
//             entry = current;
//             break;
//         }
        
//         if (current->probe_distance < probe_distance) {
//             RobinHoodEntry temp = *current;
//             *current = (RobinHoodEntry){
//                 .key_hash = hash,
//                 .probe_distance = probe_distance,
//                 .key = {0},
//             };
//             strncpy(current->key, key, MAX_KEY_LENGTH - 1);
//             entry = &temp;
//             break;
//         }
        
//         index = (index + 1) & global_cache->mask;
//         probe_distance++;
//     }

//     off_t file_offset = global_cache->file_size;
    
//     if (file_offset + value_length > global_cache->max_file_size) {
//         release_lock();
//         elog(ERROR, ERROR_MSG_SPACE);
//         PG_RETURN_BOOL(false);
//     }

//     memcpy(
//         (char*)global_cache->mapped_data + file_offset, 
//         value, 
//         value_length
//     );

//     entry->key_hash = hash;
//     entry->probe_distance = probe_distance;
//     strncpy(entry->key, key, MAX_KEY_LENGTH - 1);
//     entry->value_offset = file_offset;
//     entry->value_length = value_length;

//     global_cache->file_size += value_length;
//     global_cache->size++;

//     release_lock();
//     PG_RETURN_BOOL(true);
// }

// // Get cache entry with safe memory handling
// PG_FUNCTION_INFO_V1(ultra_cache_get);
// Datum ultra_cache_get(PG_FUNCTION_ARGS) {
//     if (PG_ARGISNULL(0) || !global_cache)
//         PG_RETURN_NULL();

//     text* key_text = PG_GETARG_TEXT_P(0);
//     char* key = text_to_cstring(key_text);
//     XXH64_hash_t hash = fast_hash(key);

//     acquire_lock(false);

//     RobinHoodEntry* entry = find_cache_entry(global_cache, key, hash);
//     if (!entry || !validate_entry(entry)) {
//         release_lock();
//         PG_RETURN_NULL();
//     }

//     text* result = (text*) palloc(VARHDRSZ + entry->value_length);
//     SET_VARSIZE(result, VARHDRSZ + entry->value_length);
    
//     memcpy(VARDATA(result), 
//            (char*)global_cache->mapped_data + entry->value_offset,
//            entry->value_length);

//     release_lock();
//     PG_RETURN_TEXT_P(result);
// }

// // Delete cache entry
// PG_FUNCTION_INFO_V1(ultra_cache_delete);
// Datum ultra_cache_delete(PG_FUNCTION_ARGS) {
//     if (PG_ARGISNULL(0) || !global_cache)
//         PG_RETURN_BOOL(false);

//     text* key_text = PG_GETARG_TEXT_P(0);
//     char* key = text_to_cstring(key_text);
//     XXH64_hash_t hash = fast_hash(key);

//     acquire_lock(true);

//     size_t index = hash & global_cache->mask;
//     uint32_t probe_distance = 0;

//     while (true) {
//         RobinHoodEntry* entry = &global_cache->entries[index];
        
//         if (entry->key[0] == '\0') {
//             release_lock();
//             PG_RETURN_BOOL(false);
//         }
        
//         if (entry->key_hash == hash && strcmp(entry->key, key) == 0) {
//             memset(entry, 0, sizeof(RobinHoodEntry));
//             global_cache->size--;
//             release_lock();
//             PG_RETURN_BOOL(true);
//         }
        
//         if (entry->probe_distance < probe_distance) {
//             release_lock();
//             PG_RETURN_BOOL(false);
//         }
        
//         index = (index + 1) & global_cache->mask;
//         probe_distance++;
//     }
// }

// // Clear entire cache
// PG_FUNCTION_INFO_V1(ultra_cache_clear);
// Datum ultra_cache_clear(PG_FUNCTION_ARGS) {
//     if (!global_cache)
//         PG_RETURN_BOOL(false);

//     acquire_lock(true);

//     memset(global_cache->entries, 0, 
//            global_cache->capacity * sizeof(RobinHoodEntry));
//     global_cache->size = 0;
//     global_cache->file_size = 0;

//     if (ftruncate(global_cache->fd, 0) != 0) {
//         release_lock();
//         PG_RETURN_BOOL(false);
//     }

//     release_lock();
//     PG_RETURN_BOOL(true);
// }

// // Module initialization
// void _PG_init(void) {
//     /* Initialize the shared memory lock if needed */
//     if (!IsUnderPostmaster) {
//         RequestAddinShmemSpace(sizeof(LWLock));
//         RequestNamedLWLockTranche("UltraCache", 1);
//         before_shmem_exit(cache_shmem_startup, 0);
//     }
// }

// // Module cleanup
// void _PG_fini(void) {
//     if (global_cache) {
//         if (global_cache->mapped_data) {
//             munmap(global_cache->mapped_data, global_cache->max_file_size);
//         }
//         if (global_cache->fd != -1) {
//             close(global_cache->fd);
//         }
//         if (global_cache->entries) {
//             pfree(global_cache->entries);
//         }
//         pfree(global_cache);
//         global_cache = NULL;
//     }
    
//     if (cache_memory_context) {
//         MemoryContextDelete(cache_memory_context);
//         cache_memory_context = NULL;
//     }
// }























//bestest one
// #include <postgres.h>
// #include "cache.h"
// #include <fmgr.h>
// #include <utils/memutils.h>
// #include <utils/builtins.h>
// #include <lz4.h>
// #include <varatt.h>
// #include <xxhash.h>  // Using XXH3 for faster, more efficient hashing

// PG_MODULE_MAGIC;

// // Optimized configuration constants
// #define INITIAL_CAPACITY 16384  // Power of 2 for faster modulo operations
// #define MAX_KEY_LENGTH 256
// #define MAX_LOAD_FACTOR 0.75
// #define SMALL_VALUE_THRESHOLD 512  // Threshold for compression
// #define CACHE_ALIGNMENT 64  // Cache line alignment for better performance

// // Compact cache entry structure with alignment and performance in mind
// typedef struct {
//     char key[MAX_KEY_LENGTH] __attribute__((aligned(CACHE_ALIGNMENT)));
//     void* value;
//     size_t value_length;
//     size_t original_length;
//     XXH64_hash_t hash;
//     bool is_compressed;
// } OptimizedCacheEntry;

// typedef struct {
//     OptimizedCacheEntry* entries;
//     size_t size;
//     size_t capacity;
//     size_t mask;  // Capacity - 1 for faster modulo
// } OptimizedHashTable;

// static OptimizedHashTable* global_cache = NULL;
// static MemoryContext cache_memory_context = NULL;

// // Extremely fast XXH3 based hash function
// static inline XXH64_hash_t fast_hash_function(const char* key) {
//     return XXH3_64bits(key, strlen(key));
// }

// // Optimized compression with minimal overhead
// static bool compress_value(const char* value, size_t length, 
//                            char** compressed, size_t* compressed_length) {
//     if (!value || length == 0) return false;

//     int max_dst_size = LZ4_compressBound(length);
//     *compressed = palloc(max_dst_size);
    
//     int result = LZ4_compress_default(value, *compressed, length, max_dst_size);
    
//     if (result <= 0) {
//         pfree(*compressed);
//         return false;
//     }

//     *compressed_length = result;
//     return true;
// }

// // Resize with power-of-two capacity for faster operations
// static void resize_hash_table(OptimizedHashTable* table) {
//     size_t new_capacity = table->capacity * 2;
//     size_t new_mask = new_capacity - 1;
//     OptimizedCacheEntry* new_entries = palloc0(new_capacity * sizeof(OptimizedCacheEntry));
    
//     for (size_t i = 0; i < table->capacity; i++) {
//         if (table->entries[i].key[0] != '\0') {
//             size_t index = table->entries[i].hash & new_mask;
            
//             while (new_entries[index].key[0] != '\0') {
//                 index = (index + 1) & new_mask;
//             }
            
//             memcpy(&new_entries[index], &table->entries[i], sizeof(OptimizedCacheEntry));
//         }
//     }
    
//     pfree(table->entries);
//     table->entries = new_entries;
//     table->capacity = new_capacity;
//     table->mask = new_mask;
// }

// // Find cache entry with linear probing
// static OptimizedCacheEntry* find_cache_entry(OptimizedHashTable* cache, 
//                                              const char* key, XXH64_hash_t hash) {
//     if (!cache) return NULL;

//     size_t index = hash & cache->mask;
//     size_t original_index = index;

//     do {
//         if (cache->entries[index].key[0] == '\0') 
//             return NULL;
        
//         if (cache->entries[index].hash == hash && 
//             strcmp(cache->entries[index].key, key) == 0) {
//             return &cache->entries[index];
//         }
        
//         index = (index + 1) & cache->mask;
//     } while (index != original_index);
    
//     return NULL;
// }

// // Initialize cache with optimized memory allocation
// static void initialize_cache() {
//     if (global_cache) return;

//     cache_memory_context = AllocSetContextCreate(
//         TopMemoryContext,
//         "UltraFastCache Memory Context",
//         ALLOCSET_DEFAULT_SIZES
//     );

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_memory_context);
    
//     global_cache = palloc0(sizeof(OptimizedHashTable));
//     global_cache->capacity = INITIAL_CAPACITY;
//     global_cache->mask = INITIAL_CAPACITY - 1;
//     global_cache->entries = palloc0(INITIAL_CAPACITY * sizeof(OptimizedCacheEntry));
//     global_cache->size = 0;
    
//     MemoryContextSwitchTo(oldcontext);
// }

// // Set cache entry with intelligent compression
// PG_FUNCTION_INFO_V1(ultra_cache_set);
// Datum ultra_cache_set(PG_FUNCTION_ARGS) {
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) 
//         PG_RETURN_BOOL(false);

//     initialize_cache();

//     text* key_text = PG_GETARG_TEXT_P(0);
//     text* value_text = PG_GETARG_TEXT_P(1);
//     char* key = text_to_cstring(key_text);
//     char* value = text_to_cstring(value_text);
//     size_t value_length = VARSIZE_ANY_EXHDR(value_text);

//     // Resize if load factor exceeds threshold
//     if ((float)global_cache->size / global_cache->capacity > MAX_LOAD_FACTOR) {
//         resize_hash_table(global_cache);
//     }

//     XXH64_hash_t hash = fast_hash_function(key);
//     OptimizedCacheEntry* entry = find_cache_entry(global_cache, key, hash);
    
//     // Find or create entry
//     if (!entry) {
//         size_t index = hash & global_cache->mask;
//         while (global_cache->entries[index].key[0] != '\0') {
//             index = (index + 1) & global_cache->mask;
//         }
        
//         entry = &global_cache->entries[index];
//         global_cache->size++;
//     } else {
//         // Free existing value
//         if (entry->value) pfree(entry->value);
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_memory_context);

//     // Compression for larger values
//     if (value_length > SMALL_VALUE_THRESHOLD) {
//         char* compressed_value = NULL;
//         size_t compressed_length = 0;
        
//         if (compress_value(value, value_length, &compressed_value, &compressed_length)) {
//             entry->value = compressed_value;
//             entry->value_length = compressed_length;
//             entry->original_length = value_length;  
//             entry->is_compressed = true;
//         } else {
//             // Fallback if compression fails
//             entry->value = palloc(value_length);
//             memcpy(entry->value, value, value_length);
//             entry->value_length = value_length;
//             entry->original_length = value_length; 
//             entry->is_compressed = false;
//         }
//     } else {
//         // Direct storage for small values
//         entry->value = palloc(value_length);
//         memcpy(entry->value, value, value_length);
//         entry->value_length = value_length;
//         entry->original_length = value_length; 
//         entry->is_compressed = false;
//     }

//     // Store key and hash
//     strncpy(entry->key, key, MAX_KEY_LENGTH - 1);
//     entry->key[MAX_KEY_LENGTH - 1] = '\0';
//     entry->hash = hash;

//     MemoryContextSwitchTo(oldcontext);

//     PG_RETURN_BOOL(true);
// }

// // Get cache entry with optimized retrieval
// PG_FUNCTION_INFO_V1(ultra_cache_get);
// Datum ultra_cache_get(PG_FUNCTION_ARGS) {
//     if (PG_ARGISNULL(0) || !global_cache)
//         PG_RETURN_NULL();

//     text* key_text = PG_GETARG_TEXT_P(0);
//     char* key = text_to_cstring(key_text);
//     XXH64_hash_t hash = fast_hash_function(key);

//     OptimizedCacheEntry* entry = find_cache_entry(global_cache, key, hash);
//     if (!entry) PG_RETURN_NULL();

//     char* result_value;
//     size_t result_length;

//     if (entry->is_compressed) {
//         result_value = palloc(entry->original_length);
//         result_length = LZ4_decompress_safe(
//             entry->value, 
//             result_value, 
//             entry->value_length, 
//             entry->original_length
//         );
//     } else {
//         result_value = entry->value;
//         result_length = entry->value_length;
//     }

//     text* result = palloc(VARHDRSZ + result_length);
//     SET_VARSIZE(result, VARHDRSZ + result_length);
//     memcpy(VARDATA(result), result_value, result_length);

//     if (entry->is_compressed) {
//         pfree(result_value);
//     }

//     PG_RETURN_TEXT_P(result);
// }

// // Delete cache entry
// PG_FUNCTION_INFO_V1(ultra_cache_delete);
// Datum ultra_cache_delete(PG_FUNCTION_ARGS) {
//     if (PG_ARGISNULL(0) || !global_cache)
//         PG_RETURN_BOOL(false);

//     text* key_text = PG_GETARG_TEXT_P(0);
//     char* key = text_to_cstring(key_text);
//     XXH64_hash_t hash = fast_hash_function(key);

//     OptimizedCacheEntry* entry = find_cache_entry(global_cache, key, hash);
//     if (!entry) PG_RETURN_BOOL(false);

//     if (entry->value) pfree(entry->value);

//     memset(entry, 0, sizeof(OptimizedCacheEntry));
//     global_cache->size--;

//     PG_RETURN_BOOL(true);
// }

// // Clear entire cache
// PG_FUNCTION_INFO_V1(ultra_cache_clear);
// Datum ultra_cache_clear(PG_FUNCTION_ARGS) {
//     if (!global_cache)
//         PG_RETURN_BOOL(false);

//     for (size_t i = 0; i < global_cache->capacity; i++) {
//         if (global_cache->entries[i].value)
//             pfree(global_cache->entries[i].value);
//     }

//     memset(global_cache->entries, 0, global_cache->capacity * sizeof(OptimizedCacheEntry));
//     global_cache->size = 0;

//     PG_RETURN_BOOL(true);
// }




























// #include <postgres.h>
// #include <fmgr.h>
// #include <utils/memutils.h>
// #include <utils/builtins.h>
// #include <lz4.h>
// #include <varatt.h>
// #include <xxhash.h>

// PG_MODULE_MAGIC;

// // Optimized configuration constants
// #define INITIAL_CAPACITY 16384  // Power of 2 for faster modulo operations
// #define MAX_KEY_LENGTH 1024
// #define MAX_LOAD_FACTOR 0.75
// #define SMALL_VALUE_THRESHOLD 4096  // Threshold for compression
// #define CACHE_ALIGNMENT 64  // Cache line alignment for better performance

// // Optimized cache entry for unlimited size entries
// typedef struct {
//     XXH64_hash_t hash;
//     char* key;  // Dynamically allocated key
//     void* value;
//     size_t value_length;
//     size_t original_length;
//     uint8_t probe_length;
//     bool is_compressed;
// } OptimizedCacheEntry;

// typedef struct {
//     OptimizedCacheEntry* entries;
//     size_t size;
//     size_t capacity;
//     size_t mask;  // Capacity - 1 for faster modulo
// } OptimizedHashTable;

// static OptimizedHashTable* global_cache = NULL;
// static MemoryContext cache_memory_context = NULL;

// // Extremely fast XXH3 based hash function
// static inline XXH64_hash_t fast_hash_function(const char* key) {
//     return XXH3_64bits(key, strlen(key));
// }

// // Dynamic memory-safe compression
// static bool compress_value(const char* value, size_t length, 
//                            char** compressed, size_t* compressed_length) {
//     if (!value || length == 0) return false;

//     int max_dst_size = LZ4_compressBound(length);
//     *compressed = palloc(max_dst_size);
    
//     int result = LZ4_compress_default(value, *compressed, length, max_dst_size);
    
//     if (result <= 0) {
//         pfree(*compressed);
//         return false;
//     }

//     *compressed_length = result;
//     return true;
// }

// // Resize with Robin Hood Hashing probing
// static void resize_hash_table(OptimizedHashTable* table) {
//     size_t new_capacity = table->capacity * 2;
//     size_t new_mask = new_capacity - 1;
//     OptimizedCacheEntry* new_entries = palloc0(new_capacity * sizeof(OptimizedCacheEntry));
    
//     for (size_t i = 0; i < table->capacity; i++) {
//         if (table->entries[i].key != NULL) {
//             OptimizedCacheEntry current = table->entries[i];
//             size_t index = current.hash & new_mask;
//             uint8_t probe_length = 0;

//             while (true) {
//                 OptimizedCacheEntry* target = &new_entries[index];
                
//                 if (target->key == NULL) {
//                     current.probe_length = probe_length;
//                     memcpy(target, &current, sizeof(OptimizedCacheEntry));
//                     break;
//                 }

//                 if (target->probe_length < probe_length) {
//                     // Swap entries to maintain Robin Hood property
//                     OptimizedCacheEntry temp = *target;
//                     *target = current;
//                     current = temp;
//                     probe_length = target->probe_length;
//                 }

//                 index = (index + 1) & new_mask;
//                 probe_length++;
//             }
//         }
//     }
    
//     pfree(table->entries);
//     table->entries = new_entries;
//     table->capacity = new_capacity;
//     table->mask = new_mask;
// }

// // Find cache entry with Robin Hood Hashing
// static OptimizedCacheEntry* find_cache_entry(OptimizedHashTable* cache, 
//                                              const char* key, XXH64_hash_t hash) {
//     if (!cache) return NULL;

//     size_t index = hash & cache->mask;
//     uint8_t probe_length = 0;

//     while (probe_length < cache->capacity) {
//         OptimizedCacheEntry* entry = &cache->entries[index];
        
//         if (entry->key == NULL) 
//             return NULL;
        
//         if (entry->hash == hash && strcmp(entry->key, key) == 0) {
//             return entry;
//         }

//         // Robin Hood: Check if current entry's probe length is less than our current probe
//         if (entry->probe_length < probe_length) {
//             return NULL;
//         }
        
//         index = (index + 1) & cache->mask;
//         probe_length++;
//     }
    
//     return NULL;
// }

// // Initialize cache with optimized memory allocation
// static void initialize_cache() {
//     if (global_cache) return;

//     cache_memory_context = AllocSetContextCreate(
//         TopMemoryContext,
//         "RobinHoodCache Memory Context",
//         ALLOCSET_DEFAULT_SIZES
//     );

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_memory_context);
    
//     global_cache = palloc0(sizeof(OptimizedHashTable));
//     global_cache->capacity = INITIAL_CAPACITY;
//     global_cache->mask = INITIAL_CAPACITY - 1;
//     global_cache->entries = palloc0(INITIAL_CAPACITY * sizeof(OptimizedCacheEntry));
//     global_cache->size = 0;
    
//     MemoryContextSwitchTo(oldcontext);
// }

// // Set cache entry with intelligent compression and Robin Hood insertion
// PG_FUNCTION_INFO_V1(ultra_cache_set);
// Datum ultra_cache_set(PG_FUNCTION_ARGS) {
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) 
//         PG_RETURN_BOOL(false);

//     initialize_cache();

//     text* key_text = PG_GETARG_TEXT_P(0);
//     text* value_text = PG_GETARG_TEXT_P(1);
//     char* key = text_to_cstring(key_text);
//     char* value = text_to_cstring(value_text);
//     size_t value_length = VARSIZE_ANY_EXHDR(value_text);

//     // Resize if load factor exceeds threshold
//     if ((float)global_cache->size / global_cache->capacity > MAX_LOAD_FACTOR) {
//         resize_hash_table(global_cache);
//     }

//     XXH64_hash_t hash = fast_hash_function(key);
//     OptimizedCacheEntry* existing_entry = find_cache_entry(global_cache, key, hash);
    
//     // If entry exists, free existing value
//     if (existing_entry) {
//         if (existing_entry->value) pfree(existing_entry->value);
//         if (existing_entry->key) pfree(existing_entry->key);
//     } else {
//         // Robin Hood insertion
//         size_t index = hash & global_cache->mask;
//         OptimizedCacheEntry new_entry = {0};
//         new_entry.hash = hash;
//         new_entry.key = palloc(strlen(key) + 1);
//         strcpy(new_entry.key, key);

//         uint8_t probe_length = 0;
//         while (probe_length < global_cache->capacity) {
//             OptimizedCacheEntry* target = &global_cache->entries[index];
            
//             if (target->key == NULL) {
//                 new_entry.probe_length = probe_length;
//                 memcpy(target, &new_entry, sizeof(OptimizedCacheEntry));
//                 global_cache->size++;
//                 break;
//             }

//             if (target->probe_length < probe_length) {
//                 // Swap entries to maintain Robin Hood property
//                 OptimizedCacheEntry temp = *target;
//                 *target = new_entry;
//                 new_entry = temp;
//                 probe_length = target->probe_length;
//             }

//             index = (index + 1) & global_cache->mask;
//             probe_length++;
//         }

//         existing_entry = &global_cache->entries[index];
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_memory_context);

//     // Compression for larger values
//     if (value_length > SMALL_VALUE_THRESHOLD) {
//         char* compressed_value = NULL;
//         size_t compressed_length = 0;
        
//         if (compress_value(value, value_length, &compressed_value, &compressed_length)) {
//             existing_entry->value = compressed_value;
//             existing_entry->value_length = compressed_length;
//             existing_entry->original_length = value_length;  
//             existing_entry->is_compressed = true;
//         } else {
//             // Fallback if compression fails
//             existing_entry->value = palloc(value_length);
//             memcpy(existing_entry->value, value, value_length);
//             existing_entry->value_length = value_length;
//             existing_entry->original_length = value_length; 
//             existing_entry->is_compressed = false;
//         }
//     } else {
//         // Direct storage for small values
//         existing_entry->value = palloc(value_length);
//         memcpy(existing_entry->value, value, value_length);
//         existing_entry->value_length = value_length;
//         existing_entry->original_length = value_length; 
//         existing_entry->is_compressed = false;
//     }

//     MemoryContextSwitchTo(oldcontext);

//     PG_RETURN_BOOL(true);
// }

// // Get cache entry with optimized retrieval
// PG_FUNCTION_INFO_V1(ultra_cache_get);
// Datum ultra_cache_get(PG_FUNCTION_ARGS) {
//     if (PG_ARGISNULL(0) || !global_cache)
//         PG_RETURN_NULL();

//     text* key_text = PG_GETARG_TEXT_P(0);
//     char* key = text_to_cstring(key_text);
//     XXH64_hash_t hash = fast_hash_function(key);

//     OptimizedCacheEntry* entry = find_cache_entry(global_cache, key, hash);
//     if (!entry) PG_RETURN_NULL();

//     char* result_value;
//     size_t result_length;

//     if (entry->is_compressed) {
//         result_value = palloc(entry->original_length);
//         result_length = LZ4_decompress_safe(
//             entry->value, 
//             result_value, 
//             entry->value_length, 
//             entry->original_length
//         );

//         if (result_length <= 0) {
//             pfree(result_value);
//             PG_RETURN_NULL();
//         }
//     } else {
//         result_value = entry->value;
//         result_length = entry->value_length;
//     }

//     text* result = palloc(VARHDRSZ + result_length);
//     SET_VARSIZE(result, VARHDRSZ + result_length);
//     memcpy(VARDATA(result), result_value, result_length);

//     if (entry->is_compressed) {
//         pfree(result_value);
//     }

//     PG_RETURN_TEXT_P(result);
// }

// // Delete cache entry
// PG_FUNCTION_INFO_V1(ultra_cache_delete);
// Datum ultra_cache_delete(PG_FUNCTION_ARGS) {
//     if (PG_ARGISNULL(0) || !global_cache)
//         PG_RETURN_BOOL(false);

//     text* key_text = PG_GETARG_TEXT_P(0);
//     char* key = text_to_cstring(key_text);
//     XXH64_hash_t hash = fast_hash_function(key);

//     OptimizedCacheEntry* entry = find_cache_entry(global_cache, key, hash);
//     if (!entry) PG_RETURN_BOOL(false);

//     // Free dynamically allocated memory
//     if (entry->value) pfree(entry->value);
//     if (entry->key) pfree(entry->key);

//     // Shift entries to maintain Robin Hood property
//     size_t index = entry - global_cache->entries;
//     size_t next_index = (index + 1) & global_cache->mask;
    
//     while (global_cache->entries[next_index].key != NULL && 
//            global_cache->entries[next_index].probe_length > 0) {
//         memcpy(&global_cache->entries[index], &global_cache->entries[next_index], 
//                sizeof(OptimizedCacheEntry));
//         global_cache->entries[next_index].key = NULL;
//         global_cache->entries[next_index].value = NULL;
        
//         index = next_index;
//         next_index = (index + 1) & global_cache->mask;
//     }

//     global_cache->size--;

//     PG_RETURN_BOOL(true);
// }

// // Clear entire cache
// PG_FUNCTION_INFO_V1(ultra_cache_clear);
// Datum ultra_cache_clear(PG_FUNCTION_ARGS) {
//     if (!global_cache)
//         PG_RETURN_BOOL(false);

//     for (size_t i = 0; i < global_cache->capacity; i++) {
//         if (global_cache->entries[i].value)
//             pfree(global_cache->entries[i].value);
//         if (global_cache->entries[i].key)
//             pfree(global_cache->entries[i].key);
//     }

//     memset(global_cache->entries, 0, global_cache->capacity * sizeof(OptimizedCacheEntry));
//     global_cache->size = 0;

//     PG_RETURN_BOOL(true);
// }






































//best one

// #include <postgres.h>
// #include "cache.h"
// #include <fmgr.h>
// #include <utils/memutils.h>
// #include <utils/builtins.h>
// #include <lz4.h>
// #include <varatt.h>
// #include <xxhash.h>  // Using XXH3 for faster, more efficient hashing

// PG_MODULE_MAGIC;

// // Optimized configuration constants
// #define INITIAL_CAPACITY 16384  // Power of 2 for faster modulo operations
// #define MAX_KEY_LENGTH 256
// #define MAX_LOAD_FACTOR 0.75
// #define SMALL_VALUE_THRESHOLD 512  // Threshold for compression
// #define CACHE_ALIGNMENT 64  // Cache line alignment for better performance

// // Compact cache entry structure with alignment and performance in mind
// typedef struct {
//     char key[MAX_KEY_LENGTH] __attribute__((aligned(CACHE_ALIGNMENT)));
//     void* value;
//     size_t value_length;
//     size_t original_length;
//     XXH64_hash_t hash;
//     bool is_compressed;
// } OptimizedCacheEntry;

// typedef struct {
//     OptimizedCacheEntry* entries;
//     size_t size;
//     size_t capacity;
//     size_t mask;  // Capacity - 1 for faster modulo
// } OptimizedHashTable;

// static OptimizedHashTable* global_cache = NULL;
// static MemoryContext cache_memory_context = NULL;

// // Extremely fast XXH3 based hash function
// static inline XXH64_hash_t fast_hash_function(const char* key) {
//     return XXH3_64bits(key, strlen(key));
// }

// // Optimized compression with minimal overhead
// static bool compress_value(const char* value, size_t length, 
//                            char** compressed, size_t* compressed_length) {
//     if (!value || length == 0) return false;

//     int max_dst_size = LZ4_compressBound(length);
//     *compressed = palloc(max_dst_size);
    
//     int result = LZ4_compress_default(value, *compressed, length, max_dst_size);
    
//     if (result <= 0) {
//         pfree(*compressed);
//         return false;
//     }

//     *compressed_length = result;
//     return true;
// }

// // Resize with power-of-two capacity for faster operations
// static void resize_hash_table(OptimizedHashTable* table) {
//     size_t new_capacity = table->capacity * 2;
//     size_t new_mask = new_capacity - 1;
//     OptimizedCacheEntry* new_entries = palloc0(new_capacity * sizeof(OptimizedCacheEntry));
    
//     for (size_t i = 0; i < table->capacity; i++) {
//         if (table->entries[i].key[0] != '\0') {
//             size_t index = table->entries[i].hash & new_mask;
            
//             while (new_entries[index].key[0] != '\0') {
//                 index = (index + 1) & new_mask;
//             }
            
//             memcpy(&new_entries[index], &table->entries[i], sizeof(OptimizedCacheEntry));
//         }
//     }
    
//     pfree(table->entries);
//     table->entries = new_entries;
//     table->capacity = new_capacity;
//     table->mask = new_mask;
// }

// // Find cache entry with linear probing
// static OptimizedCacheEntry* find_cache_entry(OptimizedHashTable* cache, 
//                                              const char* key, XXH64_hash_t hash) {
//     if (!cache) return NULL;

//     size_t index = hash & cache->mask;
//     size_t original_index = index;

//     do {
//         if (cache->entries[index].key[0] == '\0') 
//             return NULL;
        
//         if (cache->entries[index].hash == hash && 
//             strcmp(cache->entries[index].key, key) == 0) {
//             return &cache->entries[index];
//         }
        
//         index = (index + 1) & cache->mask;
//     } while (index != original_index);
    
//     return NULL;
// }

// // Initialize cache with optimized memory allocation
// static void initialize_cache() {
//     if (global_cache) return;

//     cache_memory_context = AllocSetContextCreate(
//         TopMemoryContext,
//         "UltraFastCache Memory Context",
//         ALLOCSET_DEFAULT_SIZES
//     );

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_memory_context);
    
//     global_cache = palloc0(sizeof(OptimizedHashTable));
//     global_cache->capacity = INITIAL_CAPACITY;
//     global_cache->mask = INITIAL_CAPACITY - 1;
//     global_cache->entries = palloc0(INITIAL_CAPACITY * sizeof(OptimizedCacheEntry));
//     global_cache->size = 0;
    
//     MemoryContextSwitchTo(oldcontext);
// }

// // Set cache entry with intelligent compression
// PG_FUNCTION_INFO_V1(ultra_cache_set);
// Datum ultra_cache_set(PG_FUNCTION_ARGS) {
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) 
//         PG_RETURN_BOOL(false);

//     initialize_cache();

//     text* key_text = PG_GETARG_TEXT_P(0);
//     text* value_text = PG_GETARG_TEXT_P(1);
//     char* key = text_to_cstring(key_text);
//     char* value = text_to_cstring(value_text);
//     size_t value_length = VARSIZE_ANY_EXHDR(value_text);

//     // Resize if load factor exceeds threshold
//     if ((float)global_cache->size / global_cache->capacity > MAX_LOAD_FACTOR) {
//         resize_hash_table(global_cache);
//     }

//     XXH64_hash_t hash = fast_hash_function(key);
//     OptimizedCacheEntry* entry = find_cache_entry(global_cache, key, hash);
    
//     // Find or create entry
//     if (!entry) {
//         size_t index = hash & global_cache->mask;
//         while (global_cache->entries[index].key[0] != '\0') {
//             index = (index + 1) & global_cache->mask;
//         }
        
//         entry = &global_cache->entries[index];
//         global_cache->size++;
//     } else {
//         // Free existing value
//         if (entry->value) pfree(entry->value);
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_memory_context);

//     // Compression for larger values
//     if (value_length > SMALL_VALUE_THRESHOLD) {
//         char* compressed_value = NULL;
//         size_t compressed_length = 0;
        
//         if (compress_value(value, value_length, &compressed_value, &compressed_length)) {
//             entry->value = compressed_value;
//             entry->value_length = compressed_length;
//             entry->original_length = value_length;  
//             entry->is_compressed = true;
//         } else {
//             // Fallback if compression fails
//             entry->value = palloc(value_length);
//             memcpy(entry->value, value, value_length);
//             entry->value_length = value_length;
//             entry->original_length = value_length; 
//             entry->is_compressed = false;
//         }
//     } else {
//         // Direct storage for small values
//         entry->value = palloc(value_length);
//         memcpy(entry->value, value, value_length);
//         entry->value_length = value_length;
//         entry->original_length = value_length; 
//         entry->is_compressed = false;
//     }

//     // Store key and hash
//     strncpy(entry->key, key, MAX_KEY_LENGTH - 1);
//     entry->key[MAX_KEY_LENGTH - 1] = '\0';
//     entry->hash = hash;

//     MemoryContextSwitchTo(oldcontext);

//     PG_RETURN_BOOL(true);
// }

// // Get cache entry with optimized retrieval
// PG_FUNCTION_INFO_V1(ultra_cache_get);
// Datum ultra_cache_get(PG_FUNCTION_ARGS) {
//     if (PG_ARGISNULL(0) || !global_cache)
//         PG_RETURN_NULL();

//     text* key_text = PG_GETARG_TEXT_P(0);
//     char* key = text_to_cstring(key_text);
//     XXH64_hash_t hash = fast_hash_function(key);

//     OptimizedCacheEntry* entry = find_cache_entry(global_cache, key, hash);
//     if (!entry) PG_RETURN_NULL();

//     char* result_value;
//     size_t result_length;

//     if (entry->is_compressed) {
//         result_value = palloc(entry->original_length);
//         result_length = LZ4_decompress_safe(
//             entry->value, 
//             result_value, 
//             entry->value_length, 
//             entry->original_length
//         );
//     } else {
//         result_value = entry->value;
//         result_length = entry->value_length;
//     }

//     text* result = palloc(VARHDRSZ + result_length);
//     SET_VARSIZE(result, VARHDRSZ + result_length);
//     memcpy(VARDATA(result), result_value, result_length);

//     if (entry->is_compressed) {
//         pfree(result_value);
//     }

//     PG_RETURN_TEXT_P(result);
// }

// // Delete cache entry
// PG_FUNCTION_INFO_V1(ultra_cache_delete);
// Datum ultra_cache_delete(PG_FUNCTION_ARGS) {
//     if (PG_ARGISNULL(0) || !global_cache)
//         PG_RETURN_BOOL(false);

//     text* key_text = PG_GETARG_TEXT_P(0);
//     char* key = text_to_cstring(key_text);
//     XXH64_hash_t hash = fast_hash_function(key);

//     OptimizedCacheEntry* entry = find_cache_entry(global_cache, key, hash);
//     if (!entry) PG_RETURN_BOOL(false);

//     if (entry->value) pfree(entry->value);

//     memset(entry, 0, sizeof(OptimizedCacheEntry));
//     global_cache->size--;

//     PG_RETURN_BOOL(true);
// }

// // Clear entire cache
// PG_FUNCTION_INFO_V1(ultra_cache_clear);
// Datum ultra_cache_clear(PG_FUNCTION_ARGS) {
//     if (!global_cache)
//         PG_RETURN_BOOL(false);

//     for (size_t i = 0; i < global_cache->capacity; i++) {
//         if (global_cache->entries[i].value)
//             pfree(global_cache->entries[i].value);
//     }

//     memset(global_cache->entries, 0, global_cache->capacity * sizeof(OptimizedCacheEntry));
//     global_cache->size = 0;

//     PG_RETURN_BOOL(true);
// }































// #include <postgres.h>
// #include "cache.h"
// #include <fmgr.h>
// #include "utils/memutils.h"
// #include <utils/builtins.h>
// #include <lz4.h>
// #include <varatt.h>

// static RobinHoodHashTable* global_cache = NULL;
// static MemoryContext cache_memory_context = NULL;

// PG_MODULE_MAGIC;

// // High-performance, low-collision hash function
// static inline uint64_t fast_hash_function(const char* key) {
//     register uint64_t hash = 14695981039346656037ULL;
//     while (*key) {
//         hash = ((hash ^ *key++) * 1099511628211ULL);
//     }
//     return hash ^ (hash >> 33);
// }

// // Optimized compression with better error handling
// static void compress_value(const char* value, size_t length, 
//                            char** compressed, size_t* compressed_length) {
//     if (value == NULL || length == 0) {
//         *compressed = NULL;
//         *compressed_length = 0;
//         return;
//     }

//     int max_dst_size = LZ4_compressBound(length);
//     *compressed = palloc(max_dst_size);
    
//     int result = LZ4_compress_default(value, *compressed, length, max_dst_size);
    
//     if (result <= 0) {
//         pfree(*compressed);
//         *compressed = NULL;
//         *compressed_length = 0;
//         return;
//     }

//     *compressed_length = result;
// }

// // Resize hash table when load factor exceeds threshold
// static void resize_hash_table(RobinHoodHashTable* table) {
//     size_t new_capacity = table->capacity * 2;
//     CacheEntry* new_entries = palloc0(new_capacity * sizeof(CacheEntry));
    
//     for (size_t i = 0; i < table->capacity; i++) {
//         if (table->entries[i].key[0] != '\0') {
//             uint64_t hash = fast_hash_function(table->entries[i].key);
//             size_t index = hash % new_capacity;
            
//             while (new_entries[index].key[0] != '\0') {
//                 index = (index + 1) % new_capacity;
//             }
            
//             memcpy(&new_entries[index], &table->entries[i], sizeof(CacheEntry));
//         }
//     }
    
//     pfree(table->entries);
//     table->entries = new_entries;
//     table->capacity = new_capacity;
// }

// // Find cache entry using Robin Hood probing
// static CacheEntry* find_cache_entry(RobinHoodHashTable* cache, const char* key) {
//     if (cache == NULL) return NULL;

//     uint64_t hash_value = fast_hash_function(key);
//     size_t initial_index = hash_value % cache->capacity;
//     size_t index = initial_index;

//     while (cache->entries[index].key[0] != '\0') {
//         if (strcmp(cache->entries[index].key, key) == 0) {
//             return &cache->entries[index];
//         }
        
//         index = (index + 1) % cache->capacity;
        
//         // Prevent infinite loop
//         if (index == initial_index) break;
//     }
    
//     return NULL;
// }

// // Initialize cache if not already done
// static void initialize_cache() {
//     if (global_cache == NULL) {
//         cache_memory_context = AllocSetContextCreate(
//             TopMemoryContext,
//             "UltraCache Memory Context",
//             ALLOCSET_DEFAULT_SIZES
//         );

//         MemoryContext oldcontext = MemoryContextSwitchTo(cache_memory_context);
        
//         global_cache = palloc0(sizeof(RobinHoodHashTable));
//         global_cache->capacity = CACHE_TABLE_SIZE;
//         global_cache->entries = palloc0(CACHE_TABLE_SIZE * sizeof(CacheEntry));
//         global_cache->size = 0;
//         global_cache->load_factor = 0.0;
        
//         MemoryContextSwitchTo(oldcontext);
//     }
// }

// // Set cache entry
// PG_FUNCTION_INFO_V1(ultra_cache_set);
// Datum ultra_cache_set(PG_FUNCTION_ARGS) {
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) 
//         PG_RETURN_BOOL(false);

//     initialize_cache();

//     text* key_text = PG_GETARG_TEXT_P(0);
//     text* value_text = PG_GETARG_TEXT_P(1);
//     char* key = text_to_cstring(key_text);
//     char* value = text_to_cstring(value_text);
//     size_t value_length = VARSIZE_ANY_EXHDR(value_text);

//     // Resize if load factor exceeds 0.75
//     if ((float)global_cache->size / global_cache->capacity > 0.75) {
//         resize_hash_table(global_cache);
//     }

//     CacheEntry* entry = find_cache_entry(global_cache, key);
    
//     // If no existing entry, find an empty slot
//     if (entry == NULL) {
//         uint64_t hash = fast_hash_function(key);
//         size_t index = hash % global_cache->capacity;
        
//         while (global_cache->entries[index].key[0] != '\0') {
//             index = (index + 1) % global_cache->capacity;
//         }
        
//         entry = &global_cache->entries[index];
//         global_cache->size++;
//     } else {
//         // Free existing value
//         if (entry->value) 
//             pfree(entry->value);
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_memory_context);

//     // Compression logic
//     if (value_length > COMPRESSION_THRESHOLD) {
//         char* compressed_value = NULL;
//         size_t compressed_length = 0;
        
//         compress_value(value, value_length, &compressed_value, &compressed_length);
        
//         entry->value = compressed_value;
//         entry->value_length = compressed_length;
//         entry->original_value_length = value_length;  
//         entry->is_compressed = true;
//     } else {
//         entry->value = palloc(value_length + 1);
//         memcpy(entry->value, value, value_length);
//         ((char*)entry->value)[value_length] = '\0';
//         entry->value_length = value_length;
//         entry->original_value_length = value_length; 
//         entry->is_compressed = false;
//     }

//     strncpy(entry->key, key, CACHE_MAX_KEY_LENGTH - 1);
//     entry->key[CACHE_MAX_KEY_LENGTH - 1] = '\0';

//     MemoryContextSwitchTo(oldcontext);

//     PG_RETURN_BOOL(true);
// }

// // Get cache entry
// PG_FUNCTION_INFO_V1(ultra_cache_get);
// Datum ultra_cache_get(PG_FUNCTION_ARGS) {
//     if (PG_ARGISNULL(0) || global_cache == NULL) 
//         PG_RETURN_NULL();

//     text* key_text = PG_GETARG_TEXT_P(0);
//     char* key = text_to_cstring(key_text);

//     CacheEntry* entry = find_cache_entry(global_cache, key);
//     if (entry == NULL || entry->key[0] == '\0') 
//         PG_RETURN_NULL();

//     char* result_value;
//     size_t result_length;

//     if (entry->is_compressed) {
//         char* decompressed_value = palloc(entry->original_value_length);
//         size_t decompressed_length = LZ4_decompress_safe(
//             entry->value, 
//             decompressed_value, 
//             entry->value_length, 
//             entry->original_value_length
//         );
        
//         result_value = decompressed_value;
//         result_length = decompressed_length;
//     } else {
//         result_value = entry->value;
//         result_length = entry->value_length;
//     }

//     text* result = palloc(VARHDRSZ + result_length);
//     SET_VARSIZE(result, VARHDRSZ + result_length);
//     memcpy(VARDATA(result), result_value, result_length);

//     if (entry->is_compressed) {
//         pfree(result_value);
//     }

//     PG_RETURN_TEXT_P(result);
// }

// // Delete cache entry
// PG_FUNCTION_INFO_V1(ultra_cache_delete);
// Datum ultra_cache_delete(PG_FUNCTION_ARGS) {
//     if (PG_ARGISNULL(0) || global_cache == NULL)
//         PG_RETURN_BOOL(false);

//     text* key_text = PG_GETARG_TEXT_P(0);
//     char* key = text_to_cstring(key_text);

//     CacheEntry* entry = find_cache_entry(global_cache, key);
//     if (entry == NULL || entry->key[0] == '\0')
//         PG_RETURN_BOOL(false);

//     if (entry->value)
//         pfree(entry->value);

//     memset(entry, 0, sizeof(CacheEntry));
//     global_cache->size--;

//     PG_RETURN_BOOL(true);
// }

// // Clear cache
// PG_FUNCTION_INFO_V1(ultra_cache_clear);
// Datum ultra_cache_clear(PG_FUNCTION_ARGS) {
//     if (global_cache == NULL)
//         PG_RETURN_BOOL(false);

//     for (size_t i = 0; i < global_cache->capacity; i++) {
//         if (global_cache->entries[i].value)
//             pfree(global_cache->entries[i].value);
//     }

//     memset(global_cache->entries, 0, global_cache->capacity * sizeof(CacheEntry));
//     global_cache->size = 0;

//     PG_RETURN_BOOL(true);
// }


















































// #include <postgres.h>
// #include "cache.h"
// #include <fmgr.h>
// #include "utils/memutils.h"
// #include <utils/builtins.h>
// #include <lz4.h>
// #include <varatt.h>

// static UltraCache* global_cache = NULL;
// static MemoryContext cache_memory_context = NULL;

// PG_MODULE_MAGIC;

// // Hash function implementation (XXH64-inspired)
// uint64_t hash_function(const char* key) {
//      register uint64_t hash = 14695981039346656037ULL;
//     while (*key) {
//         hash = ((hash ^ *key++) * 1099511628211ULL);
//     }
//     return hash ^ (hash >> 33);
// }

// // Find cache entry using quadratic probing
// CacheEntry* find_cache_entry(UltraCache* cache, const char* key) {
//     uint64_t hash_value = hash_function(key);
//     int initial_index = hash_value % CACHE_TABLE_SIZE;
//     int index = initial_index;
//     int i = 0;

//     // Linear scan with early termination
//     while (i < CACHE_TABLE_SIZE) {
//         CacheEntry* entry = &cache->entries[index];
        
//         // Empty slot or matching key
//         if (entry->key[0] == '\0' || strcmp(entry->key, key) == 0) {
//             return entry;
//         }

//         // Use double hashing for probing
//         index = (initial_index + i * (1 + (hash_value % (CACHE_TABLE_SIZE - 1)))) % CACHE_TABLE_SIZE;
//         i++;
//     }
    
//     return NULL;  // Table is full
// }

// void compress_value(const char* value, size_t length, char** compressed, size_t* compressed_length) {
//     if (value == NULL || length == 0) {
//         *compressed = NULL;
//         *compressed_length = 0;
//         return;
//     }

//     // Pre-allocate with LZ4 recommended size
//     int max_dst_size = LZ4_compressBound(length);
//     *compressed = palloc(max_dst_size);
    
//     // Use standard LZ4 compression instead of HC version
//     int result = LZ4_compress_default(value, *compressed, length, max_dst_size);
    
//     if (result <= 0) {
//         pfree(*compressed);
//         *compressed = NULL;
//         *compressed_length = 0;
//         return;
//     }

//     *compressed_length = result;
// }

// void decompress_value(const char* compressed, size_t compressed_length, 
//                       char** decompressed, size_t* decompressed_length,
//                       size_t original_value_length)  {
// //   if (compressed == NULL || compressed_length == 0) {
// //         elog(WARNING, "Decompress: Null or zero-length compressed data");
// //         *decompressed = NULL;
// //         *decompressed_length = 0;
// //         return;
// //     }
//     // More conservative size estimation
//       // Increased buffer
    
//     *decompressed = palloc(original_value_length);
    
//     // Detailed error logging
//     int result = LZ4_decompress_safe(compressed, *decompressed, compressed_length, original_value_length);
    
//     // if (result <= 0) {
//     //    elog(ERROR, "LZ4 Decompression Failed: Compressed Length = %zu, Original Length = %zu, Result = %d", 
//     //          compressed_length, original_value_length, result);
//     //     pfree(*decompressed);
//     //     *decompressed = NULL;
//     //     *decompressed_length = 0;
//     //     return;
//     // }

//     // elog(DEBUG1, "Decompression Success: Compressed Length = %zu, Decompressed Length = %d", 
//     //      compressed_length, result);

//     *decompressed_length = result;
// }
// // Initialize cache if not already done
// static void initialize_cache() {
//     if (global_cache == NULL) {
//         // Create a dedicated memory context for cache operations
//         if (cache_memory_context == NULL) {
//             cache_memory_context = AllocSetContextCreate(
//                 TopMemoryContext,
//                 "UltraCache Memory Context",
//                 ALLOCSET_DEFAULT_SIZES
//             );
//         }

//         // Use the dedicated memory context
//         MemoryContext oldcontext = MemoryContextSwitchTo(cache_memory_context);
        
//         global_cache = palloc0(sizeof(UltraCache));
//         global_cache->entries = palloc0(CACHE_TABLE_SIZE * sizeof(CacheEntry));
//         global_cache->size = 0;
        
//         MemoryContextSwitchTo(oldcontext);
//     }
// }

// // Set cache entry
// PG_FUNCTION_INFO_V1(ultra_cache_set);
// Datum ultra_cache_set(PG_FUNCTION_ARGS) {
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) 
//         PG_RETURN_BOOL(false);

//     initialize_cache();

//     text* key_text = PG_GETARG_TEXT_P(0);
//     text* value_text = PG_GETARG_TEXT_P(1);
//     char* key = text_to_cstring(key_text);
//     char* value = text_to_cstring(value_text);
//     size_t value_length = VARSIZE_ANY_EXHDR(value_text);
    
//     // Check key length
//     // if (strlen(key) >= CACHE_MAX_KEY_LENGTH) {
//     //     elog(ERROR, "Key length exceeds maximum allowed length");
//     //     PG_RETURN_BOOL(false);
//     // }

//     CacheEntry* entry = find_cache_entry(global_cache, key);
//     //   if (entry == NULL) {
//     //     elog(WARNING, "Could not find cache entry for key");
//     //     PG_RETURN_BOOL(false);
//     // }
//     // Free existing value if it exists
//     if (entry->value) 
//         pfree(entry->value);

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_memory_context);

//     if (value_length > CACHE_THRESHOLD_SIZE) {
//          char* compressed_value = NULL;
//         size_t compressed_length = 0;
        
//         compress_value(value, value_length, &compressed_value, &compressed_length);
        
//         // if (compressed_value == NULL) {
//         //     elog(WARNING, "Compression failed for key: %s", key);
//         //     PG_RETURN_BOOL(false);
//         // }

//         entry->value = compressed_value;
//         entry->value_length = compressed_length;
//         entry->original_value_length = value_length;  
//         entry->is_compressed = true;
//     } else {
//         entry->value = palloc(value_length + 1);
//         memcpy(entry->value, value, value_length);
//         entry->value[value_length] = '\0';
//         entry->value_length = value_length;
//         entry->original_value_length = value_length; 
//         entry->is_compressed = false;
//     }

//     strncpy(entry->key, key, CACHE_MAX_KEY_LENGTH - 1);
//     entry->key[CACHE_MAX_KEY_LENGTH - 1] = '\0';
//     if (entry->key[0] == '\0')
//         global_cache->size++;

//     // elog(DEBUG1, "Setting cache entry: Key = %s, Value Length = %zu, Compression Threshold = %d", 
//     //  key, value_length, CACHE_THRESHOLD_SIZE);

//     MemoryContextSwitchTo(oldcontext);

//     PG_RETURN_BOOL(true);
// }

// // Get cache entry
// PG_FUNCTION_INFO_V1(ultra_cache_get);
// Datum ultra_cache_get(PG_FUNCTION_ARGS) {
//     // Validate inputs and cache initialization
    
//     // if (PG_ARGISNULL(0) || global_cache == NULL) {
//     //     elog(ERROR, "Cache is not initialized or key is NULL");
//     //     PG_RETURN_NULL();
//     // }

//     text* key_text = PG_GETARG_TEXT_P(0);
//     char* key = text_to_cstring(key_text);

//     // Check key length
//     // if (strlen(key) >= CACHE_MAX_KEY_LENGTH) {
//     //     elog(WARNING, "Key length exceeds maximum allowed length");
//     //     PG_RETURN_NULL();
//     // }

//     // Locate the cache entry
//     CacheEntry* entry = find_cache_entry(global_cache, key);
//     if (entry == NULL || entry->key[0] == '\0') {
//         elog(DEBUG1, "Key not found in cache: %s", key);
//         PG_RETURN_NULL();
//     }

//     // // Add detailed logging about the entry
//     // elog(DEBUG1, "Cache Entry Details for key %s:", key);
//     // elog(DEBUG1, "Is Compressed: %s", entry->is_compressed ? "Yes" : "No");
//     // elog(DEBUG1, "Value Length: %zu", entry->value_length);

//     // Handle compressed and uncompressed values
//     char* result_value;
//     size_t result_length;

//     if (entry->is_compressed) {
        
//         // elog(DEBUG1, "Attempting to decompress value for key: %s", key);
        
//         char* decompressed_value = NULL;
//         size_t decompressed_length = 0;

//         // Log the actual compressed data details
//         // elog(DEBUG1, "Compressed Value Pointer: %p", (void*)entry->value);
//         // elog(DEBUG1, "Compressed Value Length: %zu", entry->value_length);

//         decompress_value(entry->value, entry->value_length, 
//                      &decompressed_value, &decompressed_length, 
//                      entry->original_value_length);
//         // if (decompressed_value == NULL) {
//         //     elog(ERROR, "Comprehensive decompression failed for key: %s", key);
//         //     PG_RETURN_NULL();
//         // }

//         result_value = decompressed_value;
//         result_length = decompressed_length;
//     } else {
//         // elog(DEBUG1, "Returning uncompressed value for key: %s", key);
//         result_value = entry->value;
//         result_length = entry->value_length;
//     }

//     // Create a PostgreSQL text object
//     text* result = palloc(VARHDRSZ + result_length);
//     SET_VARSIZE(result, VARHDRSZ + result_length);
//     memcpy(VARDATA(result), result_value, result_length);

//     if (entry->is_compressed) {
//         pfree(result_value);  // Free decompressed memory
//     }
//     // elog(DEBUG1, "Retrieving cache entry: Key = %s, Is Compressed = %s", 
//     //  key, entry->is_compressed ? "Yes" : "No");

//     PG_RETURN_TEXT_P(result);
// }
// // Delete cache entry
// PG_FUNCTION_INFO_V1(ultra_cache_delete);
// Datum ultra_cache_delete(PG_FUNCTION_ARGS) {
//     if (PG_ARGISNULL(0) || global_cache == NULL)
//         PG_RETURN_BOOL(false);

//     text* key_text = PG_GETARG_TEXT_P(0);
//     char* key = text_to_cstring(key_text);

//     // Check key length
//     if (strlen(key) >= CACHE_MAX_KEY_LENGTH) {
//         elog(ERROR, "Key length exceeds maximum allowed length");
//         PG_RETURN_BOOL(false);
//     }

//     CacheEntry* entry = find_cache_entry(global_cache, key);
//     if (entry == NULL || entry->key[0] == '\0')
//         PG_RETURN_BOOL(false);

//     if (entry->value)
//         pfree(entry->value);

//     memset(entry, 0, sizeof(CacheEntry));
//     global_cache->size--;

//     PG_RETURN_BOOL(true);
// }

// // Clear cache
// PG_FUNCTION_INFO_V1(ultra_cache_clear);
// Datum ultra_cache_clear(PG_FUNCTION_ARGS) {
//     if (global_cache == NULL)
//         PG_RETURN_BOOL(false);

//     for (int i = 0; i < CACHE_TABLE_SIZE; i++) {
//         if (global_cache->entries[i].value)
//             pfree(global_cache->entries[i].value);
//     }

//     memset(global_cache->entries, 0, CACHE_TABLE_SIZE * sizeof(CacheEntry));
//     global_cache->size = 0;

//     PG_RETURN_BOOL(true);
// }











// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "storage/shmem.h"
// #include "utils/memutils.h"
// #include "access/hash.h"
// #include <string.h>
// #include <stdint.h>
// #include <time.h>

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// #define CACHE_ENTRIES 4096  // Power of 2 for faster modulo
// #define ENTRY_LIFETIME 3600 // 1 hour default expiry
// #define CACHE_LINE_SIZE 64  // Typical cache line size

// // Compact, cache-aligned cache entry
// typedef struct {
//     volatile uint64_t metadata;  // Stores hash, expiry, and state
//     char *key;
//     char *value;
// } UltraFastCacheEntry;

// typedef struct {
//     UltraFastCacheEntry entries[CACHE_ENTRIES];
//     MemoryContext context;
// } UltraFastCache;

// static UltraFastCache *ultra_cache = NULL;

// // Fast, inline hash function using xxHash-like approach
// static inline uint64_t fast_hash(const char *key) {
//     uint64_t hash = 0xCBF29CE484222325ULL;
//     const unsigned char *p = (const unsigned char *)key;
    
//     while (*p) {
//         hash ^= *p++;
//         hash *= 0x100000001B3ULL;
//     }
//     return hash;
// }

// // Entry state management
// #define ENTRY_EMPTY ((uint64_t)0)
// #define ENTRY_USED ((uint64_t)1)
// #define ENTRY_TOMBSTONE ((uint64_t)2)

// // Initialize ultra-fast cache
// PG_FUNCTION_INFO_V1(ultra_cache_init);
// Datum ultra_cache_init(PG_FUNCTION_ARGS)
// {
//     if (ultra_cache == NULL) {
//         ultra_cache = MemoryContextAllocZero(TopMemoryContext, sizeof(UltraFastCache));
//         ultra_cache->context = AllocSetContextCreate(TopMemoryContext, 
//                                                      "UltraFastCacheContext",
//                                                      ALLOCSET_SMALL_SIZES);
        
//         // Initialize all entries to empty state
//         for (int i = 0; i < CACHE_ENTRIES; i++) {
//             ultra_cache->entries[i].metadata = ENTRY_EMPTY;
//             ultra_cache->entries[i].key = NULL;
//             ultra_cache->entries[i].value = NULL;
//         }
//     }
//     PG_RETURN_BOOL(true);
// }

// // Ultra-fast set operation
// PG_FUNCTION_INFO_V1(ultra_cache_set);
// Datum ultra_cache_set(PG_FUNCTION_ARGS)
// {
//     if (ultra_cache == NULL || PG_ARGISNULL(0) || PG_ARGISNULL(1)) {
//         elog(ERROR, "Cache not initialized or invalid input");
//         PG_RETURN_BOOL(false);
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(ultra_cache->context);

//     PG_TRY();
//     {
//         text *key_arg = PG_GETARG_TEXT_PP(0);
//         text *value_arg = PG_GETARG_TEXT_PP(1);
        
//         char *key = text_to_cstring(key_arg);
//         char *value = text_to_cstring(value_arg);
        
//         uint64_t hash = fast_hash(key);
//         uint32_t index = hash & (CACHE_ENTRIES - 1);
        
//         UltraFastCacheEntry *entry = &ultra_cache->entries[index];
        
//         // Prepare new metadata with full hash
//         uint64_t new_metadata = 
//             ((hash & 0xFFFFFFFF) << 32) |  // Store lower 32 bits of hash 
//             ((uint32_t)(time(NULL) + ENTRY_LIFETIME)) |  // Expiry time 
//             ((uint64_t)ENTRY_USED << 62);  // Entry state
        
//         // Avoiding unnecessary atomic operations, ensuring cache line alignment
//         __sync_lock_test_and_set(&entry->metadata, new_metadata);
        
//         // Update key and value
//         if (entry->key) pfree(entry->key);
//         if (entry->value) pfree(entry->value);
        
//         entry->key = pstrdup(key);
//         entry->value = pstrdup(value);
        
//         pfree(key);
//         pfree(value);
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
//     PG_RETURN_BOOL(true);
// }

// // Ultra-fast get operation
// PG_FUNCTION_INFO_V1(ultra_cache_get);
// Datum ultra_cache_get(PG_FUNCTION_ARGS)
// {
//     if (ultra_cache == NULL || PG_ARGISNULL(0)) {
//         elog(ERROR, "Cache not initialized or invalid input");
//         PG_RETURN_NULL();
//     }

//     text *result = NULL;
//     time_t now = time(NULL);

//     MemoryContext oldcontext = MemoryContextSwitchTo(ultra_cache->context);

//     PG_TRY();
//     {
//         text *key_arg = PG_GETARG_TEXT_PP(0);
//         char *key = text_to_cstring(key_arg);
        
//         uint64_t hash = fast_hash(key);
//         uint32_t index = hash & (CACHE_ENTRIES - 1);
        
//         UltraFastCacheEntry *entry = &ultra_cache->entries[index];
//         uint64_t metadata = entry->metadata;
        
//         // Precise hash comparison using lower 32 bits
//         if (entry->key && 
//             ((metadata >> 32) == (hash & 0xFFFFFFFF)) && 
//             ((metadata & (0x3ULL << 62)) == ((uint64_t)ENTRY_USED << 62)) &&
//             ((uint32_t)metadata > now) &&
//             strcmp(entry->key, key) == 0) 
//         {
//             result = cstring_to_text(entry->value);
//         }

//         pfree(key);
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
    
//     if (result) {
//         PG_RETURN_TEXT_P(result);
//     }
//     PG_RETURN_NULL();
// }

// // Vacuum expired entries
// PG_FUNCTION_INFO_V1(ultra_cache_vacuum);
// Datum ultra_cache_vacuum(PG_FUNCTION_ARGS)
// {
//     if (ultra_cache == NULL) {
//         elog(ERROR, "Cache not initialized");
//         PG_RETURN_INT32(0);
//     }

//     int removed_count = 0;
//     time_t now = time(NULL);

//     MemoryContext oldcontext = MemoryContextSwitchTo(ultra_cache->context);

//     PG_TRY();
//     {
//         for (int i = 0; i < CACHE_ENTRIES; i++) {
//             UltraFastCacheEntry *entry = &ultra_cache->entries[i];
//             uint64_t metadata = entry->metadata;
            
//             // Check if entry is expired
//             if (((metadata & (0x3ULL << 62)) == ((uint64_t)ENTRY_USED << 62)) && 
//                 ((uint32_t)metadata <= now)) {
                
//                 // Clear the entry
//                 if (entry->key) {
//                     pfree(entry->key);
//                     entry->key = NULL;
//                 }
                
//                 if (entry->value) {
//                     pfree(entry->value);
//                     entry->value = NULL;
//                 }
                
//                 // Reset metadata
//                 entry->metadata = ENTRY_EMPTY;
//                 removed_count++;
//             }
//         }
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
    
//     PG_RETURN_INT32(removed_count);
// }

// // Clean up cache
// PG_FUNCTION_INFO_V1(ultra_cache_cleanup);
// Datum ultra_cache_cleanup(PG_FUNCTION_ARGS)
// {
//     if (ultra_cache == NULL) {
//         PG_RETURN_BOOL(true);
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(TopMemoryContext);

//     PG_TRY();
//     {
//         for (int i = 0; i < CACHE_ENTRIES; i++) {
//             UltraFastCacheEntry *entry = &ultra_cache->entries[i];
//             if (entry->key) pfree(entry->key);
//             if (entry->value) pfree(entry->value);
//         }
        
//         MemoryContextDelete(ultra_cache->context);
//         pfree(ultra_cache);
//         ultra_cache = NULL;
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
//     PG_RETURN_BOOL(true);
// }








// second best one

// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "storage/shmem.h"
// #include "utils/memutils.h"
// #include "access/hash.h"
// #include <string.h>
// #include <stdint.h>
// #include <time.h>

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// #define CACHE_ENTRIES 4096       // Power of 2 for faster modulo
// #define ENTRY_LIFETIME 3600      // 1 hour default expiry
// #define MAX_KEY_LENGTH 64        // Reduced key length for faster comparison
// #define MAX_VALUE_LENGTH 256     // Compact value storage

// // Hyper-optimized cache entry
// typedef struct {
//     uint64_t hash;               // Precomputed hash
//     uint64_t expiry;             // Expiry timestamp
//     char key[MAX_KEY_LENGTH];    // Compact key storage
//     char value[MAX_VALUE_LENGTH]; // Compact value storage
//     bool valid;                  // Valid flag
// } __attribute__((packed)) UltraFastCacheEntry;

// // Cache structure with minimal overhead
// typedef struct {
//     UltraFastCacheEntry entries[CACHE_ENTRIES];
//     uint64_t last_access;        // Timestamp of last cache operation
// } UltraOptimizedCache;

// // Global cache instance - placed in a performance-critical memory region
// static UltraOptimizedCache *ultra_cache = NULL;

// // Extremely fast hash function (xxHash-inspired)
// static inline __attribute__((always_inline)) uint64_t 
// ultra_fast_hash(const char *key) {
//     uint64_t h = 0xCBF29CE484222325ULL;
    
//     while (*key) {
//         h ^= *key++;
//         h *= 0x100000001B3ULL;
//         h ^= h >> 33;
//     }
    
//     return h;
// }

// // Rapid key comparison - minimizes branch prediction
// static inline __attribute__((always_inline)) int 
// ultra_fast_strcmp(const char *a, const char *b) {
//     // Unrolled comparison with early exit
//     return memcmp(a, b, MAX_KEY_LENGTH);
// }

// // Initialize ultra-fast cache
// PG_FUNCTION_INFO_V1(ultra_cache_init);
// Datum ultra_cache_init(PG_FUNCTION_ARGS)
// {
//     if (ultra_cache == NULL) {
//         // Allocate cache in TopMemoryContext for persistence
//         ultra_cache = MemoryContextAllocZero(TopMemoryContext, sizeof(UltraOptimizedCache));
        
//         // Preemptively zero all entries and set valid to false
//         for (int i = 0; i < CACHE_ENTRIES; i++) {
//             ultra_cache->entries[i].valid = false;
//         }
        
//         // Set initial last access timestamp
//         ultra_cache->last_access = time(NULL);
//     }
    
//     PG_RETURN_BOOL(true);
// }
// // Ultra-fast set operation
// PG_FUNCTION_INFO_V1(ultra_cache_set);
// Datum ultra_cache_set(PG_FUNCTION_ARGS)
// {
//     if (ultra_cache == NULL || PG_ARGISNULL(0) || PG_ARGISNULL(1)) {
//         elog(ERROR, "Cache not initialized or invalid input");
//         PG_RETURN_BOOL(false);
//     }

//     // Get input arguments
//     text *key_arg = PG_GETARG_TEXT_PP(0);
//     text *value_arg = PG_GETARG_TEXT_PP(1);
    
//     // Convert to C strings
//     char *key = text_to_cstring(key_arg);
//     char *value = text_to_cstring(value_arg);
    
//     // Compute hash
//     uint64_t hash = ultra_fast_hash(key);
    
//     // Find an available slot or replace existing entry
//     bool set_success = false;
//     for (int i = 0; i < CACHE_ENTRIES; i++) {
//         UltraFastCacheEntry *entry = &ultra_cache->entries[i];
        
//         // Replace if same key exists or slot is empty
//         if (!entry->valid || 
//             (entry->hash == hash && strcmp(entry->key, key) == 0)) 
//         {
//             // Compact copy with length check
//             strncpy(entry->key, key, MAX_KEY_LENGTH - 1);
//             entry->key[MAX_KEY_LENGTH - 1] = '\0';
            
//             strncpy(entry->value, value, MAX_VALUE_LENGTH - 1);
//             entry->value[MAX_VALUE_LENGTH - 1] = '\0';
            
//             // Set metadata
//             entry->hash = hash;
//             entry->expiry = time(NULL) + ENTRY_LIFETIME;
//             entry->valid = true;
            
//             set_success = true;
//             break;
//         }
//     }
    
//     // Free temporary strings
//     pfree(key);
//     pfree(value);
    
//     PG_RETURN_BOOL(set_success);
// }

// PG_FUNCTION_INFO_V1(ultra_cache_get);
// Datum ultra_cache_get(PG_FUNCTION_ARGS)
// {
//     if (ultra_cache == NULL || PG_ARGISNULL(0)) {
//         elog(ERROR, "Cache not initialized or invalid input");
//         PG_RETURN_NULL();
//     }

//     // Get input key
//     text *key_arg = PG_GETARG_TEXT_PP(0);
//     char *key = text_to_cstring(key_arg);
    
//     // Compute hash
//     uint64_t hash = ultra_fast_hash(key);
    
//     // Search through entire cache (linear probing)
//     uint64_t now = time(NULL);
//     text *result = NULL;
    
//     for (int i = 0; i < CACHE_ENTRIES; i++) {
//         UltraFastCacheEntry *entry = &ultra_cache->entries[i];
        
//         // Check for valid entry with matching hash and key
//         if (entry->valid && 
//             entry->hash == hash && 
//             entry->expiry > now && 
//             strcmp(entry->key, key) == 0) 
//         {
//             // Convert to PostgreSQL text type
//             result = cstring_to_text_with_len(entry->value, strlen(entry->value));
//             break;
//         }
//     }
    
//     // Free temporary key
//     pfree(key);
    
//     // Return result
//     if (result) {
//         PG_RETURN_TEXT_P(result);
//     }
//     PG_RETURN_NULL();
// }
// // Vacuum operation (minimal overhead)
// PG_FUNCTION_INFO_V1(ultra_cache_vacuum);
// Datum ultra_cache_vacuum(PG_FUNCTION_ARGS)
// {
//     if (ultra_cache == NULL) {
//         PG_RETURN_INT32(0);
//     }

//     int removed_count = 0;
//     uint64_t now = time(NULL);

//     // Rapid vacuum without context switching
//     for (int i = 0; i < CACHE_ENTRIES; i++) {
//         if (ultra_cache->entries[i].expiry <= now) {
//             // Zero out expired entry
//             memset(&ultra_cache->entries[i], 0, sizeof(UltraFastCacheEntry));
//             removed_count++;
//         }
//     }
    
//     PG_RETURN_INT32(removed_count);
// }

// // Cleanup operation
// PG_FUNCTION_INFO_V1(ultra_cache_cleanup);
// Datum ultra_cache_cleanup(PG_FUNCTION_ARGS)
// {
//     if (ultra_cache != NULL) {
//         // Zero out entire cache
//         memset(ultra_cache, 0, sizeof(UltraOptimizedCache));
        
//         // Free memory
//         pfree(ultra_cache);
//         ultra_cache = NULL;
//     }
    
//     PG_RETURN_BOOL(true);
// }











//new large 
// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "storage/shmem.h"
// #include "utils/memutils.h"
// #include "access/hash.h"
// #include "miscadmin.h"
// #include "storage/lwlock.h"
// #include "utils/timestamp.h"
// #include "utils/varlena.h"  // Add this line to include varlena functions
// #include <string.h>
// #include <stdint.h>
// #include <time.h>
// #include "catalog/pg_type.h"


// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif


// // Cache expiry time in seconds (for example, 3600 seconds = 1 hour)
// #define CACHE_EXPIRY_TIME 3600
// // Configuration Parameters
// #define INITIAL_CAPACITY 1024
// #define LOAD_FACTOR 0.75
// #define MAX_KEY_LENGTH 256
// #define MAX_VALUE_SEGMENTS 1024
// #define SEGMENT_SIZE 4096
// #define UNUSED_PARAM(x) ((void)(x))

// // Cache Entry States
// typedef enum {
//     EMPTY,       // Slot is available
//     OCCUPIED,    // Slot contains valid data
//     DELETED      // Slot was previously used but is now marked for removal
// } EntryState;

// // Advanced Cache Entry Structure
// typedef struct {
//     char key[MAX_KEY_LENGTH];    // Key for the entry
//     char *value;                 // Dynamically allocated value
//     size_t value_length;         // Length of the value
//     uint64_t hash;               // Precomputed hash
//     uint64_t last_access;        // Timestamp of last access
//     EntryState state;            // Current state of the entry
// } AdvancedCacheEntry;

// // Main Cache Structure
// typedef struct {
//     AdvancedCacheEntry *entries;  // Dynamic array of entries
//     size_t capacity;              // Total slots in the cache
//     size_t size;                  // Current number of occupied slots
//     size_t deleted_count;         // Number of deleted entries
//     LWLock cache_lock_data;       // Lightweight lock for thread safety
//     LWLock *cache_lock;           // Pointer to lock
// } AdvancedTextCache;

// // Global cache instance
// static AdvancedTextCache *advanced_text_cache = NULL;

// // Faster, more optimized hash function (simplified MurmurHash3-like)
// static uint64_t advanced_text_hash(const char *key) {
//     uint64_t h1 = 0x5bd1e995;
//     size_t len = strlen(key);
//     const unsigned char *data = (const unsigned char *)key;
    
//     // Quick mixing of bytes
//     for (size_t i = 0; i < len; i++) {
//         h1 ^= data[i];
//         h1 *= 0x5bd1e995;
//         h1 = (h1 << 15) | (h1 >> 49);
//     }

//     h1 ^= len;
//     h1 ^= h1 >> 33;
//     h1 *= 0xff51afd7ed558ccd;
//     h1 ^= h1 >> 33;

//     return h1;
// }

// // Resize and rehash the cache (simplified)
// static bool resize_cache(AdvancedTextCache *cache, size_t new_capacity) {
//     AdvancedCacheEntry *new_entries;
//     size_t i;

//     LWLockAcquire(cache->cache_lock, LW_EXCLUSIVE);
    
//     new_entries = palloc0(new_capacity * sizeof(AdvancedCacheEntry));
    
//     // Fast rehashing with fewer probes
//     for (i = 0; i < cache->capacity; i++) {
//         if (cache->entries[i].state == OCCUPIED) {
//             uint64_t hash = advanced_text_hash(cache->entries[i].key);
//             size_t index = hash % new_capacity;
            
//             // Reduced linear probing
//             while (new_entries[index].state == OCCUPIED) {
//                 index = (index + 1) % new_capacity;
//             }
            
//             memcpy(&new_entries[index], &cache->entries[i], sizeof(AdvancedCacheEntry));
//         }
//     }
    
//     pfree(cache->entries);
//     cache->entries = new_entries;
//     cache->capacity = new_capacity;
//     cache->deleted_count = 0;
    
//     LWLockRelease(cache->cache_lock);
//     return true;
// }

// // Initialize advanced text cache
// PG_FUNCTION_INFO_V1(advanced_text_cache_init);
// Datum advanced_text_cache_init(PG_FUNCTION_ARGS)
// {
//     UNUSED_PARAM(fcinfo);
    
//     if (advanced_text_cache == NULL) {
//         // Allocate cache in TopMemoryContext
//         advanced_text_cache = (AdvancedTextCache *)MemoryContextAllocZero(TopMemoryContext, sizeof(AdvancedTextCache));
        
//         // Allocate initial entries
//         advanced_text_cache->entries = (AdvancedCacheEntry *)palloc0(INITIAL_CAPACITY * sizeof(AdvancedCacheEntry));
//         advanced_text_cache->capacity = INITIAL_CAPACITY;
//         advanced_text_cache->size = 0;
        
//         // Properly initialize lightweight lock
//         advanced_text_cache->cache_lock = &advanced_text_cache->cache_lock_data;
//         LWLockInitialize(advanced_text_cache->cache_lock, LWLockNewTrancheId());
//     }
    
//     PG_RETURN_BOOL(true);
// }

// PG_FUNCTION_INFO_V1(advanced_text_cache_set);
// Datum advanced_text_cache_set(PG_FUNCTION_ARGS)
// {
//     text *key_arg;
//     text *value_arg;
//     int key_len;
//     int value_len;
//     const char *key;
//     const char *value;
//     uint64_t hash;
//     size_t index;

//     UNUSED_PARAM(fcinfo);
    
//     if (advanced_text_cache == NULL || PG_ARGISNULL(0) || PG_ARGISNULL(1)) {
//         ereport(ERROR, 
//             (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
//              errmsg("Cache not initialized or invalid input")));
//         PG_RETURN_BOOL(false);
//     }

//     // Get input arguments with direct text pointer handling
//     key_arg = PG_GETARG_TEXT_PP(0);
//     value_arg = PG_GETARG_TEXT_PP(1);
    
//     // Directly get string length and pointer
//     key_len = VARSIZE_ANY_EXHDR(key_arg);
//     value_len = VARSIZE_ANY_EXHDR(value_arg);
//     key = VARDATA_ANY(key_arg);
//     value = VARDATA_ANY(value_arg);
    
//     LWLockAcquire(advanced_text_cache->cache_lock, LW_EXCLUSIVE);
    
//     // More aggressive resize check
//     if ((advanced_text_cache->size + advanced_text_cache->deleted_count) >= 
//         (advanced_text_cache->capacity * LOAD_FACTOR)) {
//         resize_cache(advanced_text_cache, advanced_text_cache->capacity * 2);
//     }
    
//     // Compute hash
//     hash = advanced_text_hash(key);
//     index = hash % advanced_text_cache->capacity;
    
//     // Optimized linear probing
//     while (advanced_text_cache->entries[index].state == OCCUPIED) {
//         // Quick key comparison
//         if (advanced_text_cache->entries[index].hash == hash &&
//             strncmp(advanced_text_cache->entries[index].key, key, 
//                     Min(key_len, MAX_KEY_LENGTH - 1)) == 0) {
            
//             // Free existing value if needed
//             if (advanced_text_cache->entries[index].value) {
//                 pfree(advanced_text_cache->entries[index].value);
//             }
            
//             // Allocate and copy new value
//             advanced_text_cache->entries[index].value = palloc(value_len + 1);
//             memcpy(advanced_text_cache->entries[index].value, value, value_len);
//             advanced_text_cache->entries[index].value[value_len] = '\0';
//             advanced_text_cache->entries[index].value_length = value_len;
//             advanced_text_cache->entries[index].last_access = (uint64_t)time(NULL);
            
//             LWLockRelease(advanced_text_cache->cache_lock);
//             PG_RETURN_BOOL(true);
//         }
        
//         index = (index + 1) % advanced_text_cache->capacity;
//     }
    
//     // Insert new entry
//     strncpy(advanced_text_cache->entries[index].key, key, 
//             Min(key_len, MAX_KEY_LENGTH - 1));
//     advanced_text_cache->entries[index].key[MAX_KEY_LENGTH - 1] = '\0';
    
//     advanced_text_cache->entries[index].value = palloc(value_len + 1);
//     memcpy(advanced_text_cache->entries[index].value, value, value_len);
//     advanced_text_cache->entries[index].value[value_len] = '\0';
//     advanced_text_cache->entries[index].value_length = value_len;
//     advanced_text_cache->entries[index].hash = hash;
//     advanced_text_cache->entries[index].last_access = (uint64_t)time(NULL);
//     advanced_text_cache->entries[index].state = OCCUPIED;
    
//     advanced_text_cache->size++;
    
//     LWLockRelease(advanced_text_cache->cache_lock);
    
//     PG_RETURN_BOOL(true);
// }

// // Get operation for advanced text cache
// PG_FUNCTION_INFO_V1(advanced_text_cache_get);
// Datum advanced_text_cache_get(PG_FUNCTION_ARGS)
// {
//     text *key_arg;
//     int key_len;
//     const char *key;
//     uint64_t hash;
//     size_t index;
//     text *result;

//     UNUSED_PARAM(fcinfo);
    
//     if (advanced_text_cache == NULL || PG_ARGISNULL(0)) {
//         ereport(ERROR, 
//             (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
//              errmsg("Cache not initialized or invalid input")));
//         PG_RETURN_NULL();
//     }

//     // Direct text pointer handling
//     key_arg = PG_GETARG_TEXT_PP(0);
//     key_len = VARSIZE_ANY_EXHDR(key_arg);
//     key = VARDATA_ANY(key_arg);
    
//     LWLockAcquire(advanced_text_cache->cache_lock, LW_SHARED);
    
//     // Compute hash
//     hash = advanced_text_hash(key);
//     index = hash % advanced_text_cache->capacity;
    
//     // Optimized lookup with hash and length checks
//     while (advanced_text_cache->entries[index].state != EMPTY) {
//         if (advanced_text_cache->entries[index].state == OCCUPIED &&
//             advanced_text_cache->entries[index].hash == hash &&
//             strncmp(advanced_text_cache->entries[index].key, key, 
//                     Min(key_len, MAX_KEY_LENGTH - 1)) == 0) {
            
//             // Update last access time
//             advanced_text_cache->entries[index].last_access = (uint64_t)time(NULL);
            
//             // Create PostgreSQL text result
//             result = palloc(VARHDRSZ + advanced_text_cache->entries[index].value_length);
//             SET_VARSIZE(result, VARHDRSZ + advanced_text_cache->entries[index].value_length);
//             memcpy(VARDATA(result), 
//                 advanced_text_cache->entries[index].value, 
//                 advanced_text_cache->entries[index].value_length);
            
//             LWLockRelease(advanced_text_cache->cache_lock);
//             PG_RETURN_TEXT_P(result);
//         }
        
//         index = (index + 1) % advanced_text_cache->capacity;
//     }
    
//     LWLockRelease(advanced_text_cache->cache_lock);
//     PG_RETURN_NULL();
// }

// // Vacuum operation to remove old entries
// PG_FUNCTION_INFO_V1(advanced_text_cache_vacuum);
// Datum advanced_text_cache_vacuum(PG_FUNCTION_ARGS)
// {
//     int removed_count = 0;
//     uint64_t now;
//     size_t i;

//     UNUSED_PARAM(fcinfo);
    
//     if (advanced_text_cache == NULL) {
//         PG_RETURN_INT32(0);
//     }

//     LWLockAcquire(advanced_text_cache->cache_lock, LW_EXCLUSIVE);
    
//     now = (uint64_t)time(NULL);

//     // Loop through all cache entries to find expired ones
//     for (i = 0; i < advanced_text_cache->capacity; i++) {
//         if (advanced_text_cache->entries[i].state == OCCUPIED && 
//             (now - advanced_text_cache->entries[i].last_access > CACHE_EXPIRY_TIME)) {
            
//             // Mark the entry as deleted
//             advanced_text_cache->entries[i].state = DELETED;
//             advanced_text_cache->size--;
//             removed_count++;
//         }
//     }

//     LWLockRelease(advanced_text_cache->cache_lock);
    
//     PG_RETURN_INT32(removed_count);
// }

















//newone large 

//large 
// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "storage/shmem.h"
// #include "utils/memutils.h"
// #include "access/hash.h"
// #include <string.h>
// #include <stdint.h>
// #include <time.h>

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// #define CACHE_SIZE 4096           // Must be power of 2
// #define CACHE_MASK (CACHE_SIZE - 1)
// #define ENTRY_LIFETIME 3600       // 1 hour default expiry
// #define MAX_KEY_LENGTH 32         // Reduced for faster comparisons
// #define MAX_SEGMENTS 128          // Maximum number of segments
// #define SEGMENT_LENGTH 2048       // Segment size

// // Compact, cache-friendly structure for cache entries
// typedef struct LargeTextCacheEntry {
//     uint64_t hash;                // Precomputed hash
//     uint32_t expiry;              // Entry expiry time
//     uint16_t key_len;             // Precise key length
//     uint16_t segment_index;       // Current segment index
//     uint16_t total_segments;      // Total number of segments
//     char key[MAX_KEY_LENGTH];     // Compact key
//     char segment[SEGMENT_LENGTH]; // Segment storage
// } __attribute__((packed)) LargeTextCacheEntry;

// // Hash table with direct indexing
// typedef struct {
//     LargeTextCacheEntry* table[CACHE_SIZE];
//     uint64_t last_access;
// } LargeTextCache;

// // Extremely fast, low-collision hash function
// static inline __attribute__((always_inline)) uint64_t 
// rapid_hash(const char *key, size_t len) {
//     uint64_t hash = 14695981039346656037ULL;
//     for (size_t i = 0; i < len; i++) {
//         hash ^= key[i];
//         hash *= 1099511628211ULL;
//     }
//     return hash;
// }

// // Global cache instance
// static LargeTextCache *large_text_cache = NULL;

// // Atomic cache initialization
// PG_FUNCTION_INFO_V1(large_text_cache_init);
// Datum large_text_cache_init(PG_FUNCTION_ARGS)
// {
//     if (large_text_cache == NULL) {
//         large_text_cache = MemoryContextAllocZero(TopMemoryContext, sizeof(LargeTextCache));
//         large_text_cache->last_access = time(NULL);
//     }
//     PG_RETURN_BOOL(true);
// }

// // Optimized set operation
// PG_FUNCTION_INFO_V1(large_text_cache_set);
// Datum large_text_cache_set(PG_FUNCTION_ARGS)
// {
//     text *key_arg, *value_arg;
//     char *key, *value;
//     size_t key_len, value_len;
//     uint64_t hash;
//     uint16_t total_segments;
//     uint32_t current_time;
    
//     if (large_text_cache == NULL || PG_ARGISNULL(0) || PG_ARGISNULL(1)) {
//         PG_RETURN_BOOL(false);
//     }

//     key_arg = PG_GETARG_TEXT_PP(0);
//     value_arg = PG_GETARG_TEXT_PP(1);
    
//     key = text_to_cstring(key_arg);
//     value = text_to_cstring(value_arg);
//     key_len = strlen(key);
//     value_len = strlen(value);
    
//     // Truncate if exceeds limits
//     if (key_len >= MAX_KEY_LENGTH || value_len > (MAX_SEGMENTS * SEGMENT_LENGTH)) {
//         pfree(key);
//         pfree(value);
//         PG_RETURN_BOOL(false);
//     }

//     hash = rapid_hash(key, key_len);
//     total_segments = (value_len + SEGMENT_LENGTH - 1) / SEGMENT_LENGTH;
//     current_time = (uint32_t)time(NULL);

//     // Remove existing entries for the same key
//     for (int i = 0; i < CACHE_SIZE; i++) {
//         if (large_text_cache->table[i] && 
//             large_text_cache->table[i]->hash == hash && 
//             strncmp(large_text_cache->table[i]->key, key, key_len) == 0) {
//             free(large_text_cache->table[i]);
//             large_text_cache->table[i] = NULL;
//         }
//     }

//     // Store each segment
//     for (uint16_t seg = 0; seg < total_segments; seg++) {
//         uint64_t index = (hash + seg) & CACHE_MASK;
//         LargeTextCacheEntry *entry = malloc(sizeof(LargeTextCacheEntry));
        
//         memset(entry, 0, sizeof(LargeTextCacheEntry));
//         strncpy(entry->key, key, MAX_KEY_LENGTH);
//         entry->key_len = key_len;
//         entry->hash = hash;
//         entry->expiry = current_time + ENTRY_LIFETIME;
//         entry->segment_index = seg;
//         entry->total_segments = total_segments;

//         // Copy segment
//         size_t copy_len = (seg == total_segments - 1) ? 
//             value_len - seg * SEGMENT_LENGTH : SEGMENT_LENGTH;
//         memcpy(entry->segment, value + seg * SEGMENT_LENGTH, copy_len);
        
//         // Replace existing entry or add new
//         if (large_text_cache->table[index]) {
//             free(large_text_cache->table[index]);
//         }
//         large_text_cache->table[index] = entry;
//     }

//     pfree(key);
//     pfree(value);
    
//     PG_RETURN_BOOL(true);
// }

// // Hyper-optimized retrieval
// PG_FUNCTION_INFO_V1(large_text_cache_get);
// Datum large_text_cache_get(PG_FUNCTION_ARGS)
// {
//     text *key_arg;
//     char *key;
//     size_t key_len;
//     uint64_t hash;
//     uint32_t current_time;
//     char *full_text;
//     text *result;
//     uint16_t total_segments = 0;
    
//     if (large_text_cache == NULL || PG_ARGISNULL(0)) {
//         PG_RETURN_NULL();
//     }

//     key_arg = PG_GETARG_TEXT_PP(0);
//     key = text_to_cstring(key_arg);
//     key_len = strlen(key);
    
//     // Truncate if exceeds limits
//     if (key_len >= MAX_KEY_LENGTH) {
//         pfree(key);
//         PG_RETURN_NULL();
//     }

//     hash = rapid_hash(key, key_len);
//     current_time = (uint32_t)time(NULL);
    
//     // Find the first segment and total segments
//     LargeTextCacheEntry *first_entry = NULL;
//     for (int i = 0; i < CACHE_SIZE; i++) {
//         LargeTextCacheEntry *entry = large_text_cache->table[i];
//         if (entry && entry->hash == hash && 
//             entry->expiry > current_time && 
//             strncmp(entry->key, key, entry->key_len) == 0) {
//             first_entry = entry;
//             total_segments = entry->total_segments;
//             break;
//         }
//     }

//     if (!first_entry) {
//         pfree(key);
//         PG_RETURN_NULL();
//     }

//     // Allocate memory for full text
//     full_text = palloc(MAX_SEGMENTS * SEGMENT_LENGTH + 1);
//     full_text[0] = '\0';
    
//     // Collect segments
//     for (uint16_t seg = 0; seg < total_segments; seg++) {
//         for (int i = 0; i < CACHE_SIZE; i++) {
//             LargeTextCacheEntry *entry = large_text_cache->table[i];
//             if (entry && entry->hash == hash && 
//                 entry->expiry > current_time && 
//                 strncmp(entry->key, key, entry->key_len) == 0 &&
//                 entry->segment_index == seg) {
//                 strcat(full_text, entry->segment);
//                 break;
//             }
//         }
//     }

//     result = cstring_to_text_with_len(full_text, strlen(full_text));
    
//     pfree(full_text);
//     pfree(key);
    
//     PG_RETURN_TEXT_P(result);
// }

// // Cleanup operation
// PG_FUNCTION_INFO_V1(large_text_cache_cleanup);
// Datum large_text_cache_cleanup(PG_FUNCTION_ARGS)
// {
//     if (large_text_cache != NULL) {
//         for (int i = 0; i < CACHE_SIZE; i++) {
//             if (large_text_cache->table[i]) {
//                 free(large_text_cache->table[i]);
//                 large_text_cache->table[i] = NULL;
//             }
//         }
//         pfree(large_text_cache);
//         large_text_cache = NULL;
//     }
    
//     PG_RETURN_BOOL(true);
// }

// // Minimal vacuum operation
// PG_FUNCTION_INFO_V1(large_text_cache_vacuum);
// Datum large_text_cache_vacuum(PG_FUNCTION_ARGS)
// {
//     uint32_t current_time = (uint32_t)time(NULL);
//     int removed_count = 0;

//     for (int i = 0; i < CACHE_SIZE; i++) {
//         if (large_text_cache->table[i] && 
//             large_text_cache->table[i]->expiry <= current_time) {
//             free(large_text_cache->table[i]);
//             large_text_cache->table[i] = NULL;
//             removed_count++;
//         }
//     }
    
//     PG_RETURN_INT32(removed_count);
// }











// //11:00 dec 11 latest

// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "storage/shmem.h"
// #include "utils/memutils.h"
// #include "access/hash.h"
// #include <string.h>
// #include <stdint.h>
// #include <time.h>

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif
// #define ULTRA_CACHE_SIZE 4096
// #define ULTRA_SEGMENT_LENGTH 2048
// #define ULTRA_MAX_SEGMENTS 64
// #define ULTRA_KEY_LENGTH 16

// typedef struct {
//     uint64_t hash;           // Precomputed hash
//     uint32_t expiry;         // Entry expiry time
//     uint16_t total_length;   // Total content length
//     uint16_t key_length;     // Key length
//     char key[ULTRA_KEY_LENGTH];  // Compact key storage
//     char data[ULTRA_MAX_SEGMENTS * ULTRA_SEGMENT_LENGTH];
// } __attribute__((packed)) UltraFastCacheEntry;

// static UltraFastCacheEntry ultra_cache[ULTRA_CACHE_SIZE];
// static uint32_t current_time = 0;

// // Ultra-fast, minimal hash function
// static inline uint64_t ultra_rapid_hash(const char *key, size_t len) {
//     uint64_t hash = 14695981039346656037ULL;
//     for (size_t i = 0; i < len; i++) {
//         hash ^= key[i];
//         hash *= 1099511628211ULL;
//     }
//     return hash;
// }


// // Initialization function
// PG_FUNCTION_INFO_V1(ultra_cache_init);
// Datum ultra_cache_init(PG_FUNCTION_ARGS)
// {
//     // Zero out the entire cache
//     memset(ultra_cache, 0, sizeof(ultra_cache));
//     current_time = (uint32_t)time(NULL);
    
//     PG_RETURN_BOOL(true);
// }


// // Cache set function
// PG_FUNCTION_INFO_V1(ultra_cache_set);
// Datum ultra_cache_set(PG_FUNCTION_ARGS)
// {
//     text *key_arg, *value_arg;
//     char *key, *value;
//     size_t key_len, value_len;
    
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) {
//         PG_RETURN_BOOL(false);
//     }

//     key_arg = PG_GETARG_TEXT_PP(0);
//     value_arg = PG_GETARG_TEXT_PP(1);
    
//     key = text_to_cstring(key_arg);
//     value = text_to_cstring(value_arg);
//     key_len = strlen(key);
//     value_len = strlen(value);
    
//     // Validate input sizes
//     if (key_len >= ULTRA_KEY_LENGTH || 
//         value_len > (ULTRA_MAX_SEGMENTS * ULTRA_SEGMENT_LENGTH)) {
//         pfree(key);
//         pfree(value);
//         PG_RETURN_BOOL(false);
//     }

//     uint64_t hash = ultra_rapid_hash(key, key_len);
//     uint32_t index = hash & (ULTRA_CACHE_SIZE - 1);
    
//     UltraFastCacheEntry *entry = &ultra_cache[index];
    
//     // Store entry
//     entry->hash = hash;
//     entry->expiry = current_time + 3600;  // 1-hour expiry
//     entry->total_length = value_len;
//     entry->key_length = key_len;
//     strncpy(entry->key, key, ULTRA_KEY_LENGTH);
//     memcpy(entry->data, value, value_len);
    
//     pfree(key);
//     pfree(value);
    
//     PG_RETURN_BOOL(true);
// }

// // Cache get function
// PG_FUNCTION_INFO_V1(ultra_cache_get);
// Datum ultra_cache_get(PG_FUNCTION_ARGS)
// {
//     text *key_arg;
//     char *key;
//     size_t key_len;
    
//     if (PG_ARGISNULL(0)) {
//         PG_RETURN_NULL();
//     }

//     key_arg = PG_GETARG_TEXT_PP(0);
//     key = text_to_cstring(key_arg);
//     key_len = strlen(key);
    
//     // Validate key size
//     if (key_len >= ULTRA_KEY_LENGTH) {
//         pfree(key);
//         PG_RETURN_NULL();
//     }

//     uint64_t hash = ultra_rapid_hash(key, key_len);
//     uint32_t index = hash & (ULTRA_CACHE_SIZE - 1);
    
//     UltraFastCacheEntry *entry = &ultra_cache[index];
    
//     // Check for valid, non-expired entry
//     if (entry->hash == hash && 
//         entry->expiry > current_time &&
//         entry->key_length == key_len && 
//         memcmp(entry->key, key, key_len) == 0) {
        
//         text *result = cstring_to_text_with_len(entry->data, entry->total_length);
//         pfree(key);
//         PG_RETURN_TEXT_P(result);
//     }
    
//     pfree(key);
//     PG_RETURN_NULL();
// }

// // Vacuum operation to remove expired entries
// PG_FUNCTION_INFO_V1(ultra_cache_vacuum);
// Datum ultra_cache_vacuum(PG_FUNCTION_ARGS)
// {
//     uint32_t now = (uint32_t)time(NULL);
//     int removed_count = 0;

//     for (int i = 0; i < ULTRA_CACHE_SIZE; i++) {
//         if (ultra_cache[i].expiry > 0 && ultra_cache[i].expiry <= now) {
//             // Mark entry as empty
//             ultra_cache[i].hash = 0;
//             ultra_cache[i].expiry = 0;
//             ultra_cache[i].total_length = 0;
//             ultra_cache[i].key_length = 0;
//             removed_count++;
//         }
//     }
    
//     PG_RETURN_INT32(removed_count);
// }

// // Cleanup function to completely reset the cache
// PG_FUNCTION_INFO_V1(ultra_cache_cleanup);
// Datum ultra_cache_cleanup(PG_FUNCTION_ARGS)
// {
//     // Zero out the entire cache
//     memset(ultra_cache, 0, sizeof(ultra_cache));
    
//     PG_RETURN_BOOL(true);
// }














// dec 11 11.18

// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "storage/shmem.h"
// #include "utils/memutils.h"
// #include <string.h>
// #include <stdint.h>
// #include <time.h>

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// #define MICRO_CACHE_SIZE 4096
// #define MICRO_DATA_SIZE (64 * 1024)  // 64KB per entry
// #define MICRO_ENTRY_LIFETIME 3600    // 1-hour entry lifetime

// typedef struct {
//     uint64_t hash;           // Compact hash
//     uint32_t length;         // Precise length
//     uint32_t expiry;         // Expiry timestamp
//     char data[MICRO_DATA_SIZE];
// } __attribute__((packed, aligned(64))) MicroFastCacheEntry;

// // Align to cache line for maximum performance
// static MicroFastCacheEntry micro_cache[MICRO_CACHE_SIZE] __attribute__((aligned(64)));
// static uint32_t micro_cache_current_time = 0;

// // Hyper-optimized, branchless hash function
// static inline __attribute__((always_inline)) uint64_t 
// micro_rapid_hash(const char *key, size_t len) {
//     register uint64_t hash = 14695981039346656037ULL;
//     for (register size_t i = 0; i < len; i++) {
//         hash ^= key[i];
//         hash *= 1099511628211ULL;
//     }
//     return hash;
// }

// // Initialization function
// PG_FUNCTION_INFO_V1(micro_cache_init);
// Datum micro_cache_init(PG_FUNCTION_ARGS)
// {
//     // Zero out the entire cache
//     memset(micro_cache, 0, sizeof(micro_cache));
    
//     // Set current time
//     micro_cache_current_time = (uint32_t)time(NULL);
    
//     PG_RETURN_BOOL(true);
// }

// // Vacuum operation to remove expired entries
// PG_FUNCTION_INFO_V1(micro_cache_vacuum);
// Datum micro_cache_vacuum(PG_FUNCTION_ARGS)
// {
//     register uint32_t now = (uint32_t)time(NULL);
//     register int removed_count = 0;

//     for (register int i = 0; i < MICRO_CACHE_SIZE; i++) {
//         // Remove entries that have expired
//         if (micro_cache[i].expiry > 0 && micro_cache[i].expiry <= now) {
//             micro_cache[i].hash = 0;
//             micro_cache[i].length = 0;
//             micro_cache[i].expiry = 0;
//             removed_count++;
//         }
//     }
    
//     PG_RETURN_INT32(removed_count);
// }

// // Cleanup function to completely reset the cache
// PG_FUNCTION_INFO_V1(micro_cache_cleanup);
// Datum micro_cache_cleanup(PG_FUNCTION_ARGS)
// {
//     // Zero out the entire cache
//     memset(micro_cache, 0, sizeof(micro_cache));
    
//     PG_RETURN_BOOL(true);
// }

// // Ultra-fast set function
// PG_FUNCTION_INFO_V1(micro_cache_set);
// Datum micro_cache_set(PG_FUNCTION_ARGS)
// {
//     // Prevent null arguments
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) {
//         PG_RETURN_BOOL(false);
//     }

//     register text *key_arg = PG_GETARG_TEXT_PP(0);
//     register text *value_arg = PG_GETARG_TEXT_PP(1);
    
//     register char *key = text_to_cstring(key_arg);
//     register char *value = text_to_cstring(value_arg);
    
//     register size_t key_len = strlen(key);
//     register size_t value_len = strlen(value);
    
//     // Validate input sizes
//     if (value_len > MICRO_DATA_SIZE) {
//         pfree(key);
//         pfree(value);
//         PG_RETURN_BOOL(false);
//     }
    
//     // Compute hash directly
//     register uint64_t hash = micro_rapid_hash(key, key_len);
//     register uint32_t index = hash & (MICRO_CACHE_SIZE - 1);
    
//     // Direct, branchless storage
//     MicroFastCacheEntry *entry = &micro_cache[index];
//     entry->hash = hash;
//     entry->length = value_len;
//     entry->expiry = micro_cache_current_time + MICRO_ENTRY_LIFETIME;
//     memcpy(entry->data, value, value_len);
    
//     pfree(key);
//     pfree(value);
    
//     PG_RETURN_BOOL(true);
// }

// // Ultra-fast retrieval function
// PG_FUNCTION_INFO_V1(micro_cache_get);
// Datum micro_cache_get(PG_FUNCTION_ARGS)
// {
//     // Prevent null arguments
//     if (PG_ARGISNULL(0)) {
//         PG_RETURN_NULL();
//     }

//     register text *key_arg = PG_GETARG_TEXT_PP(0);
//     register char *key = text_to_cstring(key_arg);
//     register size_t key_len = strlen(key);
    
//     // Compute hash directly
//     register uint64_t hash = micro_rapid_hash(key, key_len);
//     register uint32_t index = hash & (MICRO_CACHE_SIZE - 1);
    
//     // Branchless, cache-friendly retrieval
//     MicroFastCacheEntry *entry = &micro_cache[index];
    
//     // Validate entry: matching hash, not expired, and has content
//     if (entry->hash == hash && 
//         entry->expiry > micro_cache_current_time && 
//         entry->length > 0) {
//         text *result = cstring_to_text_with_len(entry->data, entry->length);
//         pfree(key);
//         PG_RETURN_TEXT_P(result);
//     }
    
//     pfree(key);
//     PG_RETURN_NULL();
// }


























// // dec 11 11.25 


// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "storage/shmem.h"
// #include "utils/memutils.h"
// #include "access/htup_details.h"  // Add this for text handling macros

// #include <string.h>
// #include <stdint.h>
// #include <time.h>

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// // Aggressive optimization constants
// #define MICRO_CACHE_SIZE 4096
// #define MICRO_DATA_SIZE (64 * 1024)  // 64KB per entry
// #define MICRO_ENTRY_LIFETIME 3600    // 1-hour entry lifetime

// // Packed, cache-aligned structure for maximum performance
// typedef struct {
//     uint64_t hash;           // Compact, unique identifier
//     uint32_t length;         // Precise data length
//     uint32_t expiry;         // Timestamp for cache invalidation
//     char data[MICRO_DATA_SIZE];
// } __attribute__((packed, aligned(64))) MicroFastCacheEntry;

// // Ensure cache is aligned and in its own cache line
// static __attribute__((aligned(64))) MicroFastCacheEntry micro_cache[MICRO_CACHE_SIZE];
// static volatile uint32_t micro_cache_current_time = 0;

// // Function prototypes
// static inline uint64_t micro_rapid_hash(const char *key, size_t len);
// Datum micro_cache_init(PG_FUNCTION_ARGS);
// Datum micro_cache_vacuum(PG_FUNCTION_ARGS);
// Datum micro_cache_cleanup(PG_FUNCTION_ARGS);
// Datum micro_cache_set(PG_FUNCTION_ARGS);
// Datum micro_cache_get(PG_FUNCTION_ARGS);

// // Hyper-optimized, branchless hash function with additional mixing
// static inline __attribute__((always_inline)) uint64_t 
// micro_rapid_hash(const char *key, size_t len) 
// {
//     register uint64_t hash = 14695981039346656037ULL;
    
//     // Unrolled hash computation for better instruction-level parallelism
//     for (register size_t i = 0; i < len; i++) {
//         hash ^= key[i];
//         hash *= 1099511628211ULL;
//         hash ^= hash >> 33;
//         hash *= 0xff51afd7ed558ccdULL;
//         hash ^= hash >> 33;
//     }
    
//     return hash;
// }

// // Highly optimized initialization function
// PG_FUNCTION_INFO_V1(micro_cache_init);
// Datum micro_cache_init(PG_FUNCTION_ARGS)
// {
//     // Declare variables at the top for C99 compatibility
//     bool result;

//     // Use memset for cache initialization
//     memset(micro_cache, 0, sizeof(micro_cache));
    
//     // Set current time
//     micro_cache_current_time = (uint32_t)time(NULL);
    
//     result = true;
//     PG_RETURN_BOOL(result);
// }

// // Ultra-fast vacuum with minimal branching
// PG_FUNCTION_INFO_V1(micro_cache_vacuum);
// Datum micro_cache_vacuum(PG_FUNCTION_ARGS)
// {
//     // Declare variables at the top
//     register uint32_t now = (uint32_t)time(NULL);
//     register int removed_count = 0;
//     register int i;

//     for (i = 0; i < MICRO_CACHE_SIZE; i++) {
//         // Branchless expiry check
//         if (micro_cache[i].expiry > 0 && micro_cache[i].expiry <= now) {
//             micro_cache[i].hash = 0;
//             micro_cache[i].length = 0;
//             micro_cache[i].expiry = 0;
//             removed_count++;
//         }
//     }
    
//     PG_RETURN_INT32(removed_count);
// }

// // Minimal overhead cleanup function
// PG_FUNCTION_INFO_V1(micro_cache_cleanup);
// Datum micro_cache_cleanup(PG_FUNCTION_ARGS)
// {
//     // Declare variables at the top
//     bool result;

//     memset(micro_cache, 0, sizeof(micro_cache));
//     result = true;
//     PG_RETURN_BOOL(result);
// }

// // Ultra-fast set function with minimal allocations
// PG_FUNCTION_INFO_V1(micro_cache_set);
// Datum micro_cache_set(PG_FUNCTION_ARGS)
// {
//     // Declare variables at the top
//     text *key_arg, *value_arg;
//     char *key, *value;
//     size_t key_len, value_len;
//     uint64_t hash;
//     uint32_t index;
//     MicroFastCacheEntry *entry;
//     bool result;

//     // Prevent null arguments
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) {
//         PG_RETURN_BOOL(false);
//     }

//     // Fetch text arguments
//     key_arg = PG_GETARG_TEXT_P(0);
//     value_arg = PG_GETARG_TEXT_P(1);
    
//     // Get key and value pointers
//     key = VARDATA_ANY(key_arg);
//     value = VARDATA_ANY(value_arg);
    
//     // Get lengths
//     key_len = VARSIZE_ANY_EXHDR(key_arg);
//     value_len = VARSIZE_ANY_EXHDR(value_arg);
    
//     // Validate input sizes
//     if (value_len > MICRO_DATA_SIZE) {
//         PG_RETURN_BOOL(false);
//     }
    
//     // Compute hash directly
//     hash = micro_rapid_hash(key, key_len);
//     index = hash & (MICRO_CACHE_SIZE - 1);
    
//     // Direct, branchless storage
//     entry = &micro_cache[index];
//     entry->hash = hash;
//     entry->length = value_len;
//     entry->expiry = micro_cache_current_time + MICRO_ENTRY_LIFETIME;
//     memcpy(entry->data, value, value_len);
    
//     result = true;
//     PG_RETURN_BOOL(result);
// }

// // Ultra-fast retrieval function
// PG_FUNCTION_INFO_V1(micro_cache_get);
// Datum micro_cache_get(PG_FUNCTION_ARGS)
// {
//     // Declare variables at the top
//     text *key_arg;
//     char *key;
//     size_t key_len;
//     uint64_t hash;
//     uint32_t index;
//     MicroFastCacheEntry *entry;
//     text *result_text;

//     // Prevent null arguments
//     if (PG_ARGISNULL(0)) {
//         PG_RETURN_NULL();
//     }

//     // Fetch text argument
//     key_arg = PG_GETARG_TEXT_P(0);
    
//     // Get key pointer
//     key = VARDATA_ANY(key_arg);
//     key_len = VARSIZE_ANY_EXHDR(key_arg);
    
//     // Compute hash directly
//     hash = micro_rapid_hash(key, key_len);
//     index = hash & (MICRO_CACHE_SIZE - 1);
    
//     // Branchless, cache-friendly retrieval
//     entry = &micro_cache[index];
    
//     // Validate entry: matching hash, not expired, and has content
//     if (entry->hash == hash && 
//         entry->expiry > micro_cache_current_time && 
//         entry->length > 0) {
//         result_text = cstring_to_text_with_len(entry->data, entry->length);
//         PG_RETURN_TEXT_P(result_text);
//     }
    
//     PG_RETURN_NULL();
// }

















// //dec 11 12.10

// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "storage/shmem.h"
// #include "utils/memutils.h"
// #include "access/htup_details.h"
// #include "utils/palloc.h"
// #include "utils/syscache.h"

// #include <string.h>
// #include <stdint.h>
// #include <time.h>

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// // Massive cache configuration
// #define MICRO_CACHE_SIZE 1024        // Reduced entries due to large size
// #define MAX_ENTRY_SIZE (500 * 1024 * 1024)  // 500MB max entry size
// #define MICRO_ENTRY_LIFETIME 60    // 1-hour entry lifetime
// #define ALLOCSET_HUGE_MINSIZE 1024 * 1024 
// #define ALLOCSET_HUGE_INITSIZE 1024 * 1024 
// #define ALLOCSET_HUGE_MAXSIZE 16 * 1024 * 1024 

// // Advanced cache entry structure
// typedef struct {
//     uint64_t hash;           // Unique identifier
//     uint32_t length;         // Precise data length
//     uint32_t expiry;         // Timestamp for cache invalidation
//     char *large_data;        // Dynamically allocated large data pointer
// } MicroLargeCacheEntry;

// // Global cache structure
// static MicroLargeCacheEntry large_micro_cache[MICRO_CACHE_SIZE];
// static volatile uint32_t large_micro_cache_current_time = 0;
// static MemoryContext large_cache_memory_context;

// // Function prototypes
// static inline uint64_t large_micro_rapid_hash(const char *key, size_t len);
// Datum large_micro_cache_init(PG_FUNCTION_ARGS);
// Datum large_micro_cache_vacuum(PG_FUNCTION_ARGS);
// Datum large_micro_cache_cleanup(PG_FUNCTION_ARGS);
// Datum large_micro_cache_set(PG_FUNCTION_ARGS);
// Datum large_micro_cache_get(PG_FUNCTION_ARGS);

// // Hyper-optimized hash function
// static inline __attribute__((always_inline)) uint64_t 
// large_micro_rapid_hash(const char *key, size_t len) 
// {
//     register uint64_t hash = 14695981039346656037ULL;
//     register uint64_t prime = 1099511628211ULL;
    
//     for (register size_t i = 0; i < len; i++) {
//         hash ^= key[i];
//         hash *= prime;
//         hash ^= hash >> 33;
//     }
    
//     return hash;
// }

// // Initialization function
// PG_FUNCTION_INFO_V1(large_micro_cache_init);
// Datum large_micro_cache_init(PG_FUNCTION_ARGS)
// {
//     // Create a dedicated memory context for large cache
//     large_cache_memory_context = AllocSetContextCreate(
//         TopMemoryContext,
//         "LargeMicroCacheContext",
//         ALLOCSET_HUGE_MINSIZE,
//         ALLOCSET_HUGE_INITSIZE,
//         ALLOCSET_HUGE_MAXSIZE
//     );

//     // Initialize cache entries
//     MemSet(large_micro_cache, 0, sizeof(large_micro_cache));
//     large_micro_cache_current_time = (uint32_t)time(NULL);
    
//     PG_RETURN_BOOL(true);
// }

// // Vacuum function to remove expired entries
// PG_FUNCTION_INFO_V1(large_micro_cache_vacuum);
// Datum large_micro_cache_vacuum(PG_FUNCTION_ARGS)
// {
//     register uint32_t now = (uint32_t)time(NULL);
//     register int removed_count = 0;

//     // Switch to large cache memory context
//     MemoryContext old_context = MemoryContextSwitchTo(large_cache_memory_context);

//     for (int i = 0; i < MICRO_CACHE_SIZE; i++) {
//         if (large_micro_cache[i].expiry > 0 && 
//             large_micro_cache[i].expiry <= now) {
            
//             // Free dynamically allocated memory
//             if (large_micro_cache[i].large_data) {
//                 pfree(large_micro_cache[i].large_data);
//                 large_micro_cache[i].large_data = NULL;
//             }

//             // Reset entry
//             large_micro_cache[i].hash = 0;
//             large_micro_cache[i].length = 0;
//             large_micro_cache[i].expiry = 0;
//             removed_count++;
//         }
//     }

//     // Restore previous memory context
//     MemoryContextSwitchTo(old_context);
    
//     PG_RETURN_INT32(removed_count);
// }

// // Complete cache cleanup
// PG_FUNCTION_INFO_V1(large_micro_cache_cleanup);
// Datum large_micro_cache_cleanup(PG_FUNCTION_ARGS)
// {
//     // Switch to large cache memory context
//     MemoryContext old_context = MemoryContextSwitchTo(large_cache_memory_context);

//     for (int i = 0; i < MICRO_CACHE_SIZE; i++) {
//         if (large_micro_cache[i].large_data) {
//             pfree(large_micro_cache[i].large_data);
//             large_micro_cache[i].large_data = NULL;
//         }
//     }

//     // Reset cache
//     MemSet(large_micro_cache, 0, sizeof(large_micro_cache));

//     // Restore previous memory context
//     MemoryContextSwitchTo(old_context);
    
//     PG_RETURN_BOOL(true);
// }

// // Set function for large entries
// PG_FUNCTION_INFO_V1(large_micro_cache_set);
// Datum large_micro_cache_set(PG_FUNCTION_ARGS)
// {
//     // Check for null arguments
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) {
//         PG_RETURN_BOOL(false);
//     }

//     // Fetch text arguments
//     text *key_arg = PG_GETARG_TEXT_P(0);
//     text *value_arg = PG_GETARG_TEXT_P(1);
    
//     // Get key and value details
//     char *key = VARDATA_ANY(key_arg);
//     char *value = VARDATA_ANY(value_arg);
    
//     size_t key_len = VARSIZE_ANY_EXHDR(key_arg);
//     size_t value_len = VARSIZE_ANY_EXHDR(value_arg);
    
//     // Validate input sizes
//     if (value_len > MAX_ENTRY_SIZE) {
//         ereport(ERROR,
//             (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
//              errmsg("Entry size exceeds maximum of 500MB")));
//     }

//     // Switch to large cache memory context
//     MemoryContext old_context = MemoryContextSwitchTo(large_cache_memory_context);

//     // Compute hash
//     uint64_t hash = large_micro_rapid_hash(key, key_len);
//     uint32_t index = hash % MICRO_CACHE_SIZE;
    
//     // Free existing entry if present
//     if (large_micro_cache[index].large_data) {
//         pfree(large_micro_cache[index].large_data);
//     }

//     // Allocate memory for large data
//     large_micro_cache[index].large_data = palloc(value_len);
//     memcpy(large_micro_cache[index].large_data, value, value_len);
    
//     // Update entry metadata
//     large_micro_cache[index].hash = hash;
//     large_micro_cache[index].length = value_len;
//     large_micro_cache[index].expiry = large_micro_cache_current_time + MICRO_ENTRY_LIFETIME;

//     // Restore previous memory context
//     MemoryContextSwitchTo(old_context);
    
//     PG_RETURN_BOOL(true);
// }

// // Get function for large entries
// PG_FUNCTION_INFO_V1(large_micro_cache_get);
// Datum large_micro_cache_get(PG_FUNCTION_ARGS)
// {
//     // Check for null arguments
//     if (PG_ARGISNULL(0)) {
//         PG_RETURN_NULL();
//     }

//     // Fetch text argument
//     text *key_arg = PG_GETARG_TEXT_P(0);
    
//     // Get key details
//     char *key = VARDATA_ANY(key_arg);
//     size_t key_len = VARSIZE_ANY_EXHDR(key_arg);
    
//     // Compute hash
//     uint64_t hash = large_micro_rapid_hash(key, key_len);
//     uint32_t index = hash % MICRO_CACHE_SIZE;
    
//     // Retrieve entry
//     MicroLargeCacheEntry *entry = &large_micro_cache[index];
    
//     // Validate entry
//     if (entry->hash == hash && 
//         entry->expiry > large_micro_cache_current_time && 
//         entry->length > 0) {
        
//         // Create text from large data
//         text *result_text = cstring_to_text_with_len(entry->large_data, entry->length);
//         PG_RETURN_TEXT_P(result_text);
//     }
    
//     PG_RETURN_NULL();
// }





//currently working december 12












// #include "postgres.h"
// #include "fmgr.h"
// #include "access/hash.h"
// #include "utils/memutils.h"
// #include "utils/builtins.h"  // Add for text_to_cstring
// #include <lz4.h>
// #include "varatt.h"

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// // Custom hash map structure
// typedef struct {
//     char *key;                 // Key string
//     char *compressed_value;    // Compressed data
//     size_t compressed_size;    // Size of compressed data
// } CacheEntry;

// // Global hash map to store cache entries
// static HTAB *cache_map = NULL;

// // Initialize the cache
// PG_FUNCTION_INFO_V1(ultra_micro_cache_init);
// Datum
// ultra_micro_cache_init(PG_FUNCTION_ARGS)
// {
//     HASHCTL ctl;
//     bool cache_created 
//     cache_created = false;
//     // Check if cache already exists
//     if (cache_map != NULL) {
//         PG_RETURN_BOOL(true);
//     }

//     // Hash table configuration
//     memset(&ctl, 0, sizeof(HASHCTL));
//     ctl.keysize = sizeof(char *);
//     ctl.entrysize = sizeof(CacheEntry);
//     ctl.hcxt = CurrentMemoryContext;

//     // Create hash table 
//     cache_map = hash_create("ultra_micro_cache", 
//                              1024,  // Initial size 
//                              &ctl, 
//                              HASH_ELEM | HASH_CONTEXT);

//     PG_RETURN_BOOL(true);
// }

// // Set a key-value pair in the cache
// PG_FUNCTION_INFO_V1(ultra_micro_cache_set);
// Datum
// ultra_micro_cache_set(PG_FUNCTION_ARGS)
// {
//     text *key_text;
//     text *value_text;
//     char *key;
//     char *value;
//     int value_len;
//     int max_dst_size;
//     char *compressed_value;
//     int compressed_size;
//     bool found;
//     void *hash_key;
//     CacheEntry *entry;
//     // Validate cache initialization
//     if (cache_map == NULL) {
//         ereport(ERROR,
//                 (errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
//                  errmsg("Cache not initialized. Call ultra_micro_cache_init first.")));
//     }

//     // Get text arguments
//         key_text = PG_GETARG_TEXT_P(0);
//         value_text = PG_GETARG_TEXT_P(1);

//     // Convert to C strings
//     key = text_to_cstring(key_text);
//     value = text_to_cstring(value_text);
//     value_len = VARSIZE_ANY_EXHDR(value_text);

//     // Prepare for LZ4 compression
//    max_dst_size = LZ4_compressBound(value_len);
//    compressed_value = palloc(max_dst_size);
    
//     // Compress data
//     compressed_size = LZ4_compress_default(
//         value, 
//         compressed_value, 
//         value_len, 
//         max_dst_size
//     );

//     // Handle compression failure
//     if (compressed_size <= 0) {
//         pfree(compressed_value);
//         pfree(key);
//         pfree(value);
//         ereport(ERROR,
//                 (errcode(ERRCODE_INTERNAL_ERROR),
//                  errmsg("LZ4 compression failed")));
//     }

//     // Insert or update cache entry
   
//     hash_key = &key;
//     entry = hash_search(cache_map, hash_key, HASH_ENTER, &found);
    
//     if (found) {
//         // Free existing resources
//         pfree(entry->compressed_value);
//         pfree(entry->key);
//     }

//     // Set new entry values
//     entry->key = key;
//     entry->compressed_value = compressed_value;
//     entry->compressed_size = compressed_size;

//     PG_RETURN_BOOL(true);
// }

// // Retrieve a value from the cache
// PG_FUNCTION_INFO_V1(ultra_micro_cache_get);
// Datum
// ultra_micro_cache_get(PG_FUNCTION_ARGS)
// {

//     text *key_text;
//     char *key;
//     void *hash_key;
//     CacheEntry *entry;
//     int max_decompressed_size;
//     char *decompressed_value;
//     int decompressed_size;
//     text *result;
//     // Validate cache initialization
//     if (cache_map == NULL) {
//         ereport(ERROR,
//                 (errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
//                  errmsg("Cache not initialized. Call ultra_micro_cache_init first.")));
//     }

//     // Get key argument
//     key_text = PG_GETARG_TEXT_P(0);
//    key = text_to_cstring(key_text);

//     // Search for entry
//     hash_key = &key;
//    entry = hash_search(cache_map, hash_key, HASH_FIND, NULL);
    
//     if (entry == NULL) {
//         pfree(key);
//         PG_RETURN_NULL();
//     }

//     // Prepare for decompression
//    max_decompressed_size = entry->compressed_size * 4;  // Estimate
//    decompressed_value = palloc(max_decompressed_size);
    
//     // Decompress value
//    decompressed_size = LZ4_decompress_safe(
//         entry->compressed_value, 
//         decompressed_value, 
//         entry->compressed_size, 
//         max_decompressed_size
//     );

//     // Handle decompression failure
//     if (decompressed_size <= 0) {
//         pfree(key);
//         pfree(decompressed_value);
//         ereport(ERROR,
//                 (errcode(ERRCODE_INTERNAL_ERROR),
//                  errmsg("LZ4 decompression failed")));
//     }

//     // Convert back to text
//         result = cstring_to_text_with_len(decompressed_value, decompressed_size);
    
//     pfree(key);
//     pfree(decompressed_value);

//     PG_RETURN_TEXT_P(result);
// }







// #include "postgres.h"
// #include "utils/memutils.h"
// #include "fmgr.h"
// #include "common/pg_lzcompress.h"
// #include "miscadmin.h"
// #include "utils/builtins.h"
// #include "utils/hsearch.h"
// #include "varatt.h"


// PG_MODULE_MAGIC;

// // Function declarations
// PG_FUNCTION_INFO_V1(ultra_micro_cache_init);
// PG_FUNCTION_INFO_V1(ultra_micro_cache_set);
// PG_FUNCTION_INFO_V1(ultra_micro_cache_get);
// PG_FUNCTION_INFO_V1(ultra_micro_cache_cleanup);

// // Cache table structure
// typedef struct {
//     char key[NAMEDATALEN];
//     int32 value_size;
//     int32 compressed_size;
//     char compressed_data[FLEXIBLE_ARRAY_MEMBER];
// } CacheEntry;

// static HTAB *ultra_cache = NULL;

// // Use the default strategy for pg_lz compression
// static const PGLZ_Strategy *compression_strategy = NULL;

// // Initialize the cache
// Datum ultra_micro_cache_init(PG_FUNCTION_ARGS) {
//     HASHCTL ctl;

//     if (ultra_cache) {
//         ereport(WARNING, (errmsg("Ultra Micro Cache is already initialized.")));
//         PG_RETURN_BOOL(false);
//     }

//     MemSet(&ctl, 0, sizeof(HASHCTL));
//     ctl.keysize = NAMEDATALEN;
//     ctl.entrysize = sizeof(CacheEntry);
//     ctl.hcxt = TopMemoryContext;

//     ultra_cache = hash_create("Ultra Micro Cache", 1024, &ctl, HASH_ELEM | HASH_CONTEXT);

//     if (!ultra_cache)
//         ereport(ERROR, (errmsg("Failed to initialize Ultra Micro Cache.")));

//     compression_strategy = PGLZ_strategy_default;

//     ereport(INFO, (errmsg("Ultra Micro Cache initialized successfully.")));
//     PG_RETURN_BOOL(true);
// }

// // Set a value in the cache
// Datum ultra_micro_cache_set(PG_FUNCTION_ARGS) {
//     text *key_text = PG_GETARG_TEXT_PP(0);
//     text *value_text = PG_GETARG_TEXT_PP(1);

//    char *key = text_to_cstring(key_text);
//     char *value = VARDATA_ANY(value_text);
//     int32 value_size = VARSIZE_ANY_EXHDR(value_text);

//     // Estimate maximum size for compression
//     int32 max_compressed_size = PGLZ_MAX_OUTPUT(value_size);
//     char *compressed_data = palloc(max_compressed_size);

//     int32 compressed_size = pglz_compress(value, value_size, compressed_data, compression_strategy);

//     if (compressed_size < 0) {
//         pfree(compressed_data);
//         ereport(ERROR, (errmsg("Compression failed for key: %s", key)));
//     }

//     // Insert into hash table
//     bool found;
//     CacheEntry *entry = (CacheEntry *) hash_search(ultra_cache, key, HASH_ENTER, &found);

//     if (found) {
//         ereport(INFO, (errmsg("Replacing existing entry for key: %s", key)));
//     }

//     // Copy data into entry
//     entry->value_size = value_size;
//     entry->compressed_size = compressed_size;
//     memcpy(entry->compressed_data, compressed_data, compressed_size);

//     pfree(compressed_data);

//     ereport(INFO, (errmsg("Value set successfully for key: %s", key)));
//     PG_RETURN_BOOL(true);
// }

// // Get a value from the cache
// Datum ultra_micro_cache_get(PG_FUNCTION_ARGS) {
//     text *key_text = PG_GETARG_TEXT_PP(0);
//     char *key = text_to_cstring(key_text);

//     bool found;
//     CacheEntry *entry = (CacheEntry *) hash_search(ultra_cache, key, HASH_FIND, &found);

//     if (!found) {
//         ereport(NOTICE, (errmsg("Key not found: %s", key)));
//         PG_RETURN_NULL();
//     }

//     // Allocate space for decompressed data
//     char *decompressed_data = palloc(entry->value_size);

//     if (pglz_decompress(entry->compressed_data, entry->compressed_size,
//                         decompressed_data, entry->value_size, true) < 0) {
//         pfree(decompressed_data);
//         ereport(ERROR, (errmsg("Decompression failed for key: %s", key)));
//     }

//     text *result = cstring_to_text_with_len(decompressed_data, entry->value_size);
//     pfree(decompressed_data);

//     PG_RETURN_TEXT_P(result);
// }

// // Cleanup the cache
// Datum ultra_micro_cache_cleanup(PG_FUNCTION_ARGS) {
//     if (ultra_cache) {
//         hash_destroy(ultra_cache);
//         ultra_cache = NULL;
//         ereport(INFO, (errmsg("Ultra Micro Cache destroyed.")));
//         PG_RETURN_BOOL(true);
//     }

//     ereport(WARNING, (errmsg("Ultra Micro Cache is not initialized.")));
//     PG_RETURN_BOOL(false);
// }











// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "utils/memutils.h"
// #include "utils/syscache.h"
// #include "access/htup.h"
// #include "access/htup_details.h"
// #include "access/toast_compression.h"
// #include "utils/guc.h"
// #include "utils/typcache.h"
// #include "catalog/pg_type.h"
// #include "funcapi.h"
// #include <string.h>
// #include "utils/elog.h"

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// /* Configuration Parameters */
// #define CACHE_SIZE 1024           /* Reduced for stability */
// #define CACHE_MASK (CACHE_SIZE - 1)
// #define MAX_TOAST_CHUNK_SIZE 1024 /* Threshold for TOAST compression */
// #define MAX_KEY_LENGTH 255        /* Maximum key length */
// #define ERRCODE_CACHE_FULL MAKE_SQLSTATE('C','A','0','0','1')
// /* Cache Entry Status */
// typedef enum {
//     ENTRY_EMPTY = 0,
//     ENTRY_OCCUPIED = 1,
//     ENTRY_DELETED = 2
// } EntryCacheStatus;

// /* Advanced Cache Entry with TOAST Support */
// typedef struct {
//     uint64_t hash_key;           /* Precomputed hash key */
//     char *key;                   /* Stored key for precise matching */
//     Datum toast_datum;           /* TOAST datum for storing large values */
//     Size original_size;          /* Original value size */
//     bool is_toasted;             /* Flag to indicate TOAST status */
//     EntryCacheStatus status;     /* Entry status */
// } UltraToastCacheEntry;

// /* Global TOAST-Enhanced Cache Structure */
// typedef struct {
//     UltraToastCacheEntry *entries;
//     MemoryContext cache_context;
//     int total_entries;
//     bool initialized;
// } UltraToastCache;

// /* Static global cache instance */
// static UltraToastCache UltraCache = {0};

// /* High-Performance Hash Function (FNV-1a variant) */
// static uint64_t
// ultra_hash_key(const char *key, int len)
// {
//     uint64_t hash = 14695981039346656037ULL;
//     for (int i = 0; i < len; i++) {
//         hash ^= (unsigned char)key[i];
//         hash *= 1099511628211ULL;
//     }
//     return hash;
// }

// /* Safe key comparison function */
// static bool
// safe_key_compare(const char *key1, const char *key2)
// {
//     if (!key1 || !key2) {
//         return false;
//     }
//     return strcmp(key1, key2) == 0;
// }

// /* Find next available slot in cache */
// static int
// find_cache_slot(uint64_t hash, const char *key)
// {
//     uint32_t index = hash & CACHE_MASK;
//     uint32_t original_index = index;
//     int attempts = 0;

//     while (attempts < CACHE_SIZE) {
//         UltraToastCacheEntry *entry = &UltraCache.entries[index];
        
//         /* Empty or deleted slot is available */
//         if (entry->status == ENTRY_EMPTY || entry->status == ENTRY_DELETED) {
//             return index;
//         }
        
//         /* If hash matches and keys are the same, we found the right slot */
//         if (entry->hash_key == hash && 
//             (entry->key ? safe_key_compare(entry->key, key) : false)) {
//             return index;
//         }

//         /* Linear probing */
//         index = (index + 1) & CACHE_MASK;
//         attempts++;
//     }

//     /* Cache is full */
//     ereport(ERROR, 
//         (errcode(ERRCODE_CACHE_FULL),
//          errmsg("Ultra cache is full. Cannot add more entries.")));
//     return -1;
// }

// /* Initialize TOAST-Enhanced Cache */
// PG_FUNCTION_INFO_V1(ultra_micro_cache_init);
// Datum
// ultra_micro_cache_init(PG_FUNCTION_ARGS)
// {
//     /* Prevent multiple initializations */
//     if (UltraCache.initialized) {
//         PG_RETURN_BOOL(true);
//     }

//     /* Create dedicated memory context */
//     UltraCache.cache_context = AllocSetContextCreate(
//         TopMemoryContext,
//         "UltraToastCache",
//         ALLOCSET_SMALL_SIZES
//     );

//     /* Switch to cache context */
//     MemoryContext old_context = MemoryContextSwitchTo(UltraCache.cache_context);

//     /* Allocate entries */
//     UltraCache.entries = palloc0(CACHE_SIZE * sizeof(UltraToastCacheEntry));

//     /* Initialize entries */
//     for (int i = 0; i < CACHE_SIZE; i++) {
//         UltraCache.entries[i].status = ENTRY_EMPTY;
//     }

//     /* Reset counters */
//     UltraCache.total_entries = 0;
//     UltraCache.initialized = true;

//     /* Restore previous memory context */
//     MemoryContextSwitchTo(old_context);

//     PG_RETURN_BOOL(true);
// }

// /* TOAST-Enhanced Cache Set Function */
// PG_FUNCTION_INFO_V1(ultra_micro_cache_set);
// Datum
// ultra_micro_cache_set(PG_FUNCTION_ARGS)
// {
//     /* Ensure cache is initialized */
//     if (!UltraCache.initialized) {
//         ereport(ERROR,
//             (errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
//              errmsg("Cache not initialized. Call ultra_micro_cache_init() first.")));
//     }

//     text *key_arg = PG_GETARG_TEXT_P(0);
//     text *value_arg = PG_GETARG_TEXT_P(1);
    
//     char *key = VARDATA(key_arg);
//     int32 key_len = VARSIZE(key_arg) - VARHDRSZ;
//     char safe_key[MAX_KEY_LENGTH + 1] = {0};

//     /* Safely copy key */
//     int copy_len = Min(key_len, MAX_KEY_LENGTH);
//     memcpy(safe_key, key, copy_len);

//     /* Compute hash */
//     uint64_t hash = ultra_hash_key(safe_key, copy_len);
    
//     /* Find appropriate slot */
//     int index = find_cache_slot(hash, safe_key);
    
//     /* Switch to cache memory context */
//     MemoryContext old_context = MemoryContextSwitchTo(UltraCache.cache_context);

//     /* Cleanup existing entry if needed */
//     UltraToastCacheEntry *entry = &UltraCache.entries[index];
//     if (entry->status == ENTRY_OCCUPIED) {
//         if (entry->key) {
//             pfree(entry->key);
//             entry->key = NULL;
//         }
//         if (entry->is_toasted && entry->toast_datum) {
//             pfree((void*)entry->toast_datum);
//             entry->toast_datum = 0;
//         }
//     }

//     /* Prepare TOAST datum */
//     Datum value_datum = PointerGetDatum(value_arg);
//     bool is_toasted = false;
//     Datum compressed_datum;

//     /* Attempt TOAST compression */
//     if (VARSIZE(value_arg) > MAX_TOAST_CHUNK_SIZE) {
//         compressed_datum = (Datum) pglz_compress_datum((struct varlena *) value_datum);
//         is_toasted = true;
//     } else {
//         compressed_datum = value_datum;
//     }

//     /* Store new entry */
//     entry->hash_key = hash;
//     entry->key = pstrdup(safe_key);  /* Deep copy of key */
//     entry->toast_datum = compressed_datum;
//     entry->original_size = VARSIZE(value_arg) - VARHDRSZ;
//     entry->is_toasted = is_toasted;
//     entry->status = ENTRY_OCCUPIED;

//     /* Increment total entries if this is a new slot */
//     if (UltraCache.total_entries < CACHE_SIZE) {
//         UltraCache.total_entries++;
//     }

//     /* Restore memory context */
//     MemoryContextSwitchTo(old_context);

//     PG_RETURN_BOOL(true);
// }

// /* Highly Optimized Retrieval Function with TOAST Support */
// PG_FUNCTION_INFO_V1(ultra_micro_cache_get);
// Datum
// ultra_micro_cache_get(PG_FUNCTION_ARGS)
// {
//     /* Ensure cache is initialized */
//     if (!UltraCache.initialized) {
//         ereport(ERROR,
//             (errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
//              errmsg("Cache not initialized. Call ultra_micro_cache_init() first.")));
//     }

//     text *key_arg = PG_GETARG_TEXT_P(0);
//     char *key = VARDATA(key_arg);
//     int32 key_len = VARSIZE(key_arg) - VARHDRSZ;
//     char safe_key[MAX_KEY_LENGTH + 1] = {0};

//     /* Safely copy key */
//     int copy_len = Min(key_len, MAX_KEY_LENGTH);
//     memcpy(safe_key, key, copy_len);

//     /* Compute hash */
//     uint64_t hash = ultra_hash_key(safe_key, copy_len);
//     uint32_t index = hash & CACHE_MASK;
//     int attempts = 0;

//     /* Linear probing to find the correct entry */
//     while (attempts < CACHE_SIZE) {
//         UltraToastCacheEntry *entry = &UltraCache.entries[index];

//         /* Check if this is the right entry */
//         if (entry->status == ENTRY_OCCUPIED && 
//             entry->hash_key == hash && 
//             safe_key_compare(entry->key, safe_key)) {
            
//             Datum value_datum;
//             text *result_text;

//             /* Retrieve and potentially decompress TOAST datum */
//             PG_TRY();
//             {
//                 if (entry->is_toasted) {
//                     value_datum = (Datum) pglz_decompress_datum((struct varlena *) entry->toast_datum);
//                 } else {
//                     value_datum = entry->toast_datum;
//                 }

//                 /* Convert back to text */
//                 result_text = DatumGetTextP(value_datum);
//             }
//             PG_CATCH();
//             {
//                 /* Error handling */
//                 ereport(ERROR,
//                     (errcode(ERRCODE_DATA_EXCEPTION),
//                      errmsg("Failed to retrieve or decompress cache entry")));
//             }
//             PG_END_TRY();

//             PG_RETURN_TEXT_P(result_text);
//         }

//         /* If we hit an empty slot, the key doesn't exist */
//         if (entry->status == ENTRY_EMPTY) {
//             PG_RETURN_NULL();
//         }

//         /* Linear probing */
//         index = (index + 1) & CACHE_MASK;
//         attempts++;
//     }

//     /* Key not found after searching entire cache */
//     PG_RETURN_NULL();
// }

// /* Cleanup Function with TOAST Awareness */
// PG_FUNCTION_INFO_V1(ultra_micro_cache_cleanup);
// Datum
// ultra_micro_cache_cleanup(PG_FUNCTION_ARGS)
// {
//     if (!UltraCache.initialized) {
//         PG_RETURN_BOOL(false);
//     }

//     for (int i = 0; i < CACHE_SIZE; i++) {
//         UltraToastCacheEntry *entry = &UltraCache.entries[i];
        
//         if (entry->status == ENTRY_OCCUPIED) {
//             /* Free key */
//             if (entry->key) {
//                 pfree(entry->key);
//                 entry->key = NULL;
//             }

//             /* Properly delete TOAST datum if compressed */
//             if (entry->is_toasted && entry->toast_datum) {
//                 pfree((void*)entry->toast_datum);
//                 entry->toast_datum = 0;
//             }

//             /* Reset entry */
//             entry->hash_key = 0;
//             entry->is_toasted = false;
//             entry->status = ENTRY_EMPTY;
//         }
//     }

//     /* Reset total entries count */
//     UltraCache.total_entries = 0;

//     PG_RETURN_BOOL(true);
// }

// /* Optional Shutdown Function */
// PG_FUNCTION_INFO_V1(ultra_micro_cache_shutdown);
// Datum
// ultra_micro_cache_shutdown(PG_FUNCTION_ARGS)
// {
//     if (!UltraCache.initialized) {
//         PG_RETURN_BOOL(false);
//     }

//     /* Cleanup entries */
//     ultra_micro_cache_cleanup(fcinfo);

//     /* Delete memory context */
//     MemoryContextDelete(UltraCache.cache_context);

//     /* Reset global structure */
//     UltraCache.entries = NULL;
//     UltraCache.total_entries = 0;
//     UltraCache.initialized = false;

//     PG_RETURN_BOOL(true);
// }




















/*
 * Dynamic In-Memory Cache for PostgreSQL
 * Provides unlimited, session-based key-value storage
 */
/*
 * Dynamic In-Memory Cache for PostgreSQL
 * Provides unlimited, session-based key-value storage
//  */
// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "utils/memutils.h"
// #include "utils/syscache.h"
// #include "access/htup.h"
// #include "access/htup_details.h"
// #include "utils/guc.h"
// #include "common/pg_lzcompress.h"  // PostgreSQL compression header
// #include <string.h>
// #include "funcapi.h"

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// #define INITIAL_BUCKET_COUNT 16
// #define LOAD_FACTOR 0.75
// #define MAX_KEY_LENGTH 1024

// // Configuration variables with default values
// static int ultra_cache_max_size_mb = 256;  // Default 256 MB
// static int ultra_max_entry_size_mb = 10;   // Default 10 MB per entry

// // Forward declaration of cache entry structure
// typedef struct CacheEntry CacheEntry;

// // Cache Entry Structure
// struct CacheEntry {
//     char *key;             // Dynamically allocated key
//     void *compressed_value; // Pointer to compressed value data
//     size_t compressed_value_size; // Size of compressed value
//     size_t original_value_size;   // Original uncompressed value size
//     struct CacheEntry *next; // For handling hash collisions
// };

// // Global Cache Structure
// typedef struct {
//     CacheEntry **buckets;      // Dynamic array of entry pointers
//     size_t bucket_count;       // Number of hash buckets
//     size_t entry_count;        // Total number of entries
//     size_t total_cache_size;   // Track total memory used
//     MemoryContext cache_context; // Dedicated memory context
//     bool is_initialized;       // Initialization flag
// } UltraCacheStruct;

// // Declare global cache structure
// static UltraCacheStruct UltraCache = {0};

// // Function prototypes to resolve potential circular dependencies
// static void resize_cache(void);
// static uint32_t ultra_hash(const char *key, size_t len);

// // Configuration Setup Function
// void
// _PG_init(void)
// {
//     // Define GUC variables
//     DefineCustomIntVariable(
//         "ultra_cache.max_size_mb",
//         "Maximum total cache size in megabytes",
//         "Sets the maximum memory used by the ultra micro cache",
//         &ultra_cache_max_size_mb,
//         256,    // Default
//         16,     // Minimum 
//         4096,   // Maximum (4GB)
//         PGC_USERSET,  // Context 
//         0,      // Flags
//         NULL,   // Check hook
//         NULL,   // Assign hook
//         NULL    // Show hook
//     );

//     DefineCustomIntVariable(
//         "ultra_cache.max_entry_size_mb",
//         "Maximum size of individual cache entries in megabytes",
//         "Sets the maximum size of a single cache entry",
//         &ultra_max_entry_size_mb,
//         10,     // Default 
//         1,      // Minimum
//         1024,   // Maximum (1GB per entry)
//         PGC_USERSET,  
//         0,      
//         NULL,   
//         NULL,   
//         NULL    
//     );
// }
// // Optimized Hash Function
// static uint32_t 
// ultra_hash(const char *key, size_t len) 
// {
//     uint32_t hash = 2166136261U;  // FNV-1a initial basis
//     size_t i;
//     for (i = 0; i < len; i++) {
//         hash ^= (unsigned char)key[i];
//         hash *= 16777619U;  // FNV prime
//     }
//     return hash;
// }

// // Resize and Rehash Cache
// static void
// resize_cache(void) 
// {
//     size_t new_bucket_count = UltraCache.bucket_count * 2;
//     CacheEntry **new_buckets = NULL;
//     size_t i;

//     new_buckets = palloc0(sizeof(CacheEntry*) * new_bucket_count);
    
//     // Rehash existing entries
//     for (i = 0; i < UltraCache.bucket_count; i++) {
//         CacheEntry *entry = UltraCache.buckets[i];
//         while (entry) {
//             CacheEntry *next = entry->next;
            
//             // Compute new hash index
//             uint32_t hash = ultra_hash(entry->key, strlen(entry->key));
//             size_t new_index = hash % new_bucket_count;
            
//             // Insert into new bucket
//             entry->next = new_buckets[new_index];
//             new_buckets[new_index] = entry;
            
//             entry = next;
//         }
//     }
    
//     // Free old buckets and update cache
//     pfree(UltraCache.buckets);
//     UltraCache.buckets = new_buckets;
//     UltraCache.bucket_count = new_bucket_count;
// }
// // Initialize Cache
// PG_FUNCTION_INFO_V1(ultra_micro_cache_init);
// Datum
// ultra_micro_cache_init(PG_FUNCTION_ARGS)
// {
//     MemoryContext old_context;
//     size_t initial_bucket_count;

//     // Prevent multiple initializations
//     if (UltraCache.is_initialized) 
//         PG_RETURN_BOOL(false);

//     // Create memory context
//     UltraCache.cache_context = AllocSetContextCreate(
//         TopMemoryContext,
//         "UltraDynamicCache",
//         ALLOCSET_SMALL_SIZES
//     );

//     // Switch to cache context
//     old_context = MemoryContextSwitchTo(UltraCache.cache_context);

//     // Dynamically calculate bucket count based on configured max size
//     initial_bucket_count = (ultra_cache_max_size_mb * 1024 * 1024) / 
//                            (sizeof(CacheEntry*) + sizeof(CacheEntry));
//     initial_bucket_count = Max(16, initial_bucket_count);

//     // Initialize buckets
//     UltraCache.bucket_count = initial_bucket_count;
//     UltraCache.buckets = palloc0(sizeof(CacheEntry*) * UltraCache.bucket_count);
    
//     // Switch back to previous context
//     MemoryContextSwitchTo(old_context);

//     UltraCache.is_initialized = true;
//     UltraCache.entry_count = 0;
//     UltraCache.total_cache_size = 0;

//     PG_RETURN_BOOL(true);
// }

// // Set Cache Entry
// PG_FUNCTION_INFO_V1(ultra_micro_cache_set);
// Datum
// ultra_micro_cache_set(PG_FUNCTION_ARGS)
// {
//     text *key_arg = NULL;
//     text *value_arg = NULL;
//     char *key = NULL;
//     char *value = NULL;
//     int32 key_len = 0;
//     int32 value_len = 0;
//     MemoryContext old_context;
//     uint32_t hash;
//     size_t bucket_index;
//     CacheEntry *prev = NULL;
//     CacheEntry *current = NULL;
//     CacheEntry *new_entry = NULL;
//     size_t potential_total_size;
    
//     // Compression variables
//     int32 compressed_len;
//     char *compressed_data = NULL;

//     // Ensure cache is initialized
//     if (!UltraCache.is_initialized) 
//         ultra_micro_cache_init(fcinfo);

//     // Check for null arguments
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
//         PG_RETURN_BOOL(false);

//     // Process arguments
//     key_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
//     value_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(1));

//     key = VARDATA(key_arg);
//     value = VARDATA(value_arg);
//     key_len = VARSIZE(key_arg) - VARHDRSZ;
//     value_len = VARSIZE(value_arg) - VARHDRSZ;

//     // Prepare compression buffer
//     compressed_data = palloc(PGLZ_MAX_OUTPUT(value_len));

//     // Compress the value
//     compressed_len = pglz_compress(
//         value, 
//         value_len, 
//         compressed_data, 
//         PGLZ_strategy_default
//     );
    
//     if (compressed_len < 0) {
//         // Compression failed, use original data
//         pfree(compressed_data);
//         compressed_len = value_len;
//         compressed_data = value;
//     }

//     // Calculate potential total size before using it
//     potential_total_size = UltraCache.total_cache_size + 
//                            key_len + compressed_len + sizeof(CacheEntry);

//     if (potential_total_size > (ultra_cache_max_size_mb * 1024 * 1024)) {
//         if (compressed_len != value_len) 
//             pfree(compressed_data);
//         pfree(key_arg);
//         pfree(value_arg);
//         PG_RETURN_BOOL(false);
//     }

//     // Switch to cache memory context
//     old_context = MemoryContextSwitchTo(UltraCache.cache_context);

//     // Compute hash and bucket index
//     hash = ultra_hash(key, key_len);
//     bucket_index = hash % UltraCache.bucket_count;

//     // Check for existing entry and remove if found
//     current = UltraCache.buckets[bucket_index];
//     while (current) {
//         if (strlen(current->key) == key_len && 
//             memcmp(current->key, key, key_len) == 0) {
//             // Remove existing entry
//             if (prev) 
//                 prev->next = current->next;
//             else 
//                 UltraCache.buckets[bucket_index] = current->next;
            
//             // Adjust total cache size
//             UltraCache.total_cache_size -= (current->compressed_value_size + 
//                                             strlen(current->key) + sizeof(CacheEntry));
            
//             // Free existing entry resources
//             pfree(current->key);
//             pfree(current->compressed_value);
//             pfree(current);
//             UltraCache.entry_count--;
//             break;
//         }
//         prev = current;
//         current = current->next;
//     }

//     // Create new entry
//     new_entry = palloc(sizeof(CacheEntry));
//     new_entry->key = palloc(key_len + 1);
//     memcpy(new_entry->key, key, key_len);
//     new_entry->key[key_len] = '\0';
    
//     // Store compressed value
//     new_entry->compressed_value = palloc(compressed_len);
//     memcpy(new_entry->compressed_value, compressed_data, compressed_len);
//     new_entry->compressed_value_size = compressed_len;
//     new_entry->original_value_size = value_len;

//     // Link into bucket
//     new_entry->next = UltraCache.buckets[bucket_index];
//     UltraCache.buckets[bucket_index] = new_entry;
//     UltraCache.entry_count++;

//     // Update total cache size
//     UltraCache.total_cache_size += key_len + compressed_len + sizeof(CacheEntry);

//     // Resize if load factor exceeded
//     if ((float)UltraCache.entry_count / UltraCache.bucket_count > LOAD_FACTOR) {
//         resize_cache();
//     }

//     // Restore previous memory context
//     MemoryContextSwitchTo(old_context);

//     // Free temporary buffers
//     if (compressed_len != value_len) 
//         pfree(compressed_data);
//     pfree(key_arg);
//     pfree(value_arg);

//     PG_RETURN_BOOL(true);
// }

// // Get Cache Entry with Decompression
// PG_FUNCTION_INFO_V1(ultra_micro_cache_get);
// Datum
// ultra_micro_cache_get(PG_FUNCTION_ARGS)
// {
//     text *key_arg = NULL;
//     char *key = NULL;
//     int32 key_len = 0;
//     text *result_text = NULL;
//     CacheEntry *current = NULL;
//     uint32_t hash;
//     size_t bucket_index;
    
//     // Decompression variables
//     char *decompressed_data = NULL;
//     int32 decompressed_len;

//     // Check for null or uninitialized cache
//     if (PG_ARGISNULL(0) || !UltraCache.is_initialized)
//         PG_RETURN_NULL();

//     // Process key argument
//     key_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
//     key = VARDATA(key_arg);
//     key_len = VARSIZE(key_arg) - VARHDRSZ;

//     // Compute hash and bucket index
//     hash = ultra_hash(key, key_len);
//     bucket_index = hash % UltraCache.bucket_count;

//     // Search for entry
//     current = UltraCache.buckets[bucket_index];
//     while (current) {
//         if (strlen(current->key) == key_len && 
//             memcmp(current->key, key, key_len) == 0) {
//             // Allocate buffer for decompressed data
//             decompressed_data = palloc(current->original_value_size);

//             // Decompress if needed
//             if (current->compressed_value_size != current->original_value_size) {
//                 decompressed_len = pglz_decompress(
//                     current->compressed_value, 
//                     current->compressed_value_size, 
//                     decompressed_data, 
//                     current->original_value_size,
//                     true  // check for complete decompression
//                 );
                
//                 if (decompressed_len < 0) {
//                     // Decompression failed
//                     pfree(decompressed_data);
//                     pfree(key_arg);
//                     PG_RETURN_NULL();
//                 }
//             } else {
//                 // Not compressed, just copy
//                 memcpy(decompressed_data, current->compressed_value, current->original_value_size);
//                 decompressed_len = current->original_value_size;
//             }

//             // Create text result
//             result_text = cstring_to_text_with_len(
//                 decompressed_data,
//                 decompressed_len
//             );

//             // Free temporary buffer
//             pfree(decompressed_data);
//             pfree(key_arg);
//             PG_RETURN_TEXT_P(result_text);
//         }
//         current = current->next;
//     }

//     // Not found
//     pfree(key_arg);
//     PG_RETURN_NULL();
// }
// // Cleanup function needs modification to free compressed values
// PG_FUNCTION_INFO_V1(ultra_micro_cache_cleanup);
// Datum
// ultra_micro_cache_cleanup(PG_FUNCTION_ARGS)
// {
//     size_t i;

//     if (!UltraCache.is_initialized)
//         PG_RETURN_BOOL(false);

//     // Free all entries
//     for (i = 0; i < UltraCache.bucket_count; i++) {
//         CacheEntry *current = UltraCache.buckets[i];
//         while (current) {
//             CacheEntry *next = current->next;
            
//             // Free individual entry resources
//             pfree(current->key);
//             pfree(current->compressed_value);
//             pfree(current);
            
//             current = next;
//         }
//         UltraCache.buckets[i] = NULL;
//     }

//     // Reset cache state
//     UltraCache.entry_count = 0;
//     UltraCache.total_cache_size = 0;

//     PG_RETURN_BOOL(true);
// }
// // Shutdown and Free Cache
// PG_FUNCTION_INFO_V1(ultra_micro_cache_shutdown);
// Datum
// ultra_micro_cache_shutdown(PG_FUNCTION_ARGS)
// {
//     if (!UltraCache.is_initialized)
//         PG_RETURN_BOOL(false);

//     // Cleanup entries
//     ultra_micro_cache_cleanup(fcinfo);

//     // Free buckets
//     pfree(UltraCache.buckets);

//     // Delete memory context
//     MemoryContextDelete(UltraCache.cache_context);

//     // Reset global structure
//     memset(&UltraCache, 0, sizeof(UltraCache));

//     PG_RETURN_BOOL(true);
// }

















/* 
 * PostgreSQL RAM Cache Extension
 * Provides an 
 * optimized in-memory key-value cache
 */

// Undefine conflicting macros before including simplehash

// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "utils/memutils.h"
// #include "utils/guc.h"
// #include "utils/timestamp.h"
// #include "lib/simplehash.h"
// #include "access/hash.h"

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// // Configurable parameters with default values
// static int ram_cache_max_entries = 10000;
// static int ram_cache_max_value_size = 4096;

// // Define a hash table entry with added status
// typedef struct {
//     char *key;              // Key
//     char *value;            // Value
//     size_t value_length;    // Length of value
//     TimestampTz expires_at; // Expiration timestamp
//     int status;             // Add status for simplehash compatibility
// } RamCacheEntry;


// // Macro definitions for simplehash
// #define SH_PREFIX ram_cache_hash
// #define SH_ELEMENT_TYPE RamCacheEntry
// #define SH_KEY key
// #define SH_KEY_TYPE char*
// #define SH_HASH_KEY(tb, key) hash_any((unsigned char *)key, strlen(key))
// #define SH_COMPARE_KEYS(tb, key1, key2) (strcmp(key1, key2) == 0)
// #define SH_EQUAL(tb, key1, key2) (strcmp(key1, key2) == 0)  // Add this line
// #define SH_SCOPE static inline
// #define SH_DEFINE
// #define SH_DECLARE
// #include "lib/simplehash.h"

// // Global cache management structure
// static ram_cache_hash_hash *ram_cache = NULL;
// static MemoryContext ram_cache_context = NULL;

// // Function prototypes
// Datum ram_cache_init(PG_FUNCTION_ARGS);
// Datum ram_cache_set(PG_FUNCTION_ARGS);
// Datum ram_cache_get(PG_FUNCTION_ARGS);
// Datum ram_cache_clear(PG_FUNCTION_ARGS);
// Datum ram_cache_delete(PG_FUNCTION_ARGS);

// // Declare function info
// PG_FUNCTION_INFO_V1(ram_cache_init);
// PG_FUNCTION_INFO_V1(ram_cache_set);
// PG_FUNCTION_INFO_V1(ram_cache_get);
// PG_FUNCTION_INFO_V1(ram_cache_clear);
// PG_FUNCTION_INFO_V1(ram_cache_delete);

// // Plugin initialization function to register configuration parameters
// void _PG_init(void)
// {
//     DefineCustomIntVariable(
//         "ram_cache.max_entries",
//         "Maximum number of entries in the RAM cache",
//         "Sets the total number of key-value pairs that can be stored in memory",
//         &ram_cache_max_entries,
//         10000,  // default
//         100,    // minimum
//         1000000, // maximum
//         PGC_USERSET,
//         0,
//         NULL,
//         NULL,
//         NULL
//     );

//     DefineCustomIntVariable(
//         "ram_cache.max_value_size",
//         "Maximum size of a single cache entry value",
//         "Sets the maximum allowed size for a single cache entry value in bytes",
//         &ram_cache_max_value_size,
//         4096,   // default (4KB)
//         64,     // maximum
//         104857600, // maximum (100MB)
//         PGC_USERSET,
//         0,
//         NULL,
//         NULL,
//         NULL
//     );
// }

// // Initialize RAM Cache
// Datum ram_cache_init(PG_FUNCTION_ARGS)
// {
//     MemoryContext old_context;

//     if (ram_cache != NULL) {
//         elog(NOTICE, "RAM Cache is already initialized");
//         PG_RETURN_BOOL(false);
//     }

//     // Create a dedicated memory context for the cache
//     old_context = MemoryContextSwitchTo(TopMemoryContext);
//     ram_cache_context = AllocSetContextCreate(
//         TopMemoryContext,
//         "RamCacheContext",
//         ALLOCSET_DEFAULT_SIZES
//     );
//     MemoryContextSwitchTo(ram_cache_context);

//     // Initialize hash table with correct arguments
//     ram_cache = ram_cache_hash_create(ram_cache_context, ram_cache_max_entries, NULL);

//     // Switch back to the original memory context
//     MemoryContextSwitchTo(old_context);

//     elog(NOTICE, "RAM Cache initialized with max entries: %d", ram_cache_max_entries);
//     PG_RETURN_BOOL(true);
// }

// // Set a key-value pair with optional TTL
// // Set a key-value pair with optional TTL
// Datum ram_cache_set(PG_FUNCTION_ARGS)
// {
//     text *key_arg;
//     text *value_arg;
//     char *key;
//     char *value;
//     size_t value_len;
//     TimestampTz expires_at;
//     MemoryContext old_context;
//     RamCacheEntry *entry;
//     bool local_found;
//     bool result = false;  // Initialize result to false

//     // Log entry point
//     elog(NOTICE, "Entering ram_cache_set function");

//     // Check if cache is initialized
//     if (ram_cache == NULL) {
//         elog(ERROR, "RAM Cache is not initialized. Call ram_cache_init first.");
//         PG_RETURN_BOOL(false);
//     }

//     // Validate arguments
//     if (PG_NARGS() < 2) {
//         elog(ERROR, "Insufficient arguments for ram_cache_set");
//         PG_RETURN_BOOL(false);
//     }

//     // Get arguments safely
//     key_arg = PG_GETARG_TEXT_P(0);
//     value_arg = PG_GETARG_TEXT_P(1);
    
//     // Convert to C strings
//     key = text_to_cstring(key_arg);
//     value = text_to_cstring(value_arg);
//     value_len = strlen(value);

//     // Log key and value for debugging
//     elog(NOTICE, "Setting key: %s, value: %s", key, value);

//     // Validate value size
//     if (value_len >= ram_cache_max_value_size) {
//         elog(ERROR, "Value size exceeds maximum allowed size.");
//         pfree(key);
//         pfree(value);
//         PG_RETURN_BOOL(false);
//     }

//     // Switch to RAM cache memory context
//     old_context = MemoryContextSwitchTo(ram_cache_context);

//     // Try to insert or update entry
//     entry = ram_cache_hash_insert(ram_cache, key, &local_found);
    
//     if (entry != NULL) {
//         // Set status to in-use
//         entry->status = 1;

//         // Free existing key and value if they exist
//         if (entry->key) pfree(entry->key);
//         if (entry->value) pfree(entry->value);
        
//         // Set new key and value
//         entry->key = pstrdup(key);
//         entry->value = pstrdup(value);
//         entry->value_length = value_len;
//         entry->expires_at = 0;  // No expiration by default

//         result = true;  // Successfully set
//         elog(NOTICE, "Successfully set key-value pair");
//     } else {
//         elog(NOTICE, "Failed to insert entry");
//     }

//     // Free temporary strings
//     pfree(key);
//     pfree(value);

//     // Switch back to original memory context
//     MemoryContextSwitchTo(old_context);

//     PG_RETURN_BOOL(result);
// }

// // Retrieve a value by key
// Datum ram_cache_get(PG_FUNCTION_ARGS)
// {
//     text *key_arg;
//     char *key;
//     RamCacheEntry *entry;

//     if (ram_cache == NULL) {
//         elog(ERROR, "RAM Cache is not initialized. Call ram_cache_init first.");
//         PG_RETURN_NULL();
//     }

//     key_arg = PG_GETARG_TEXT_P(0);
//     key = text_to_cstring(key_arg);

//     entry = ram_cache_hash_lookup(ram_cache, key);
    
//     // Free the temporary key string
//     pfree(key);

//     if (entry != NULL) {
//         if (entry->expires_at == 0 || entry->expires_at > GetCurrentTimestamp()) {
//             text *result = cstring_to_text_with_len(entry->value, entry->value_length);
//             PG_RETURN_TEXT_P(result);
//         }
//     }

//     PG_RETURN_NULL();
// }
// // Delete a specific key from cache
// Datum ram_cache_delete(PG_FUNCTION_ARGS)
// {
//     text *key_arg;
//     char *key;
//     RamCacheEntry *entry;

//     if (ram_cache == NULL) {
//         elog(ERROR, "RAM Cache is not initialized. Call ram_cache_init first.");
//         PG_RETURN_BOOL(false);
//     }

//     key_arg = PG_GETARG_TEXT_P(0);
//     key = text_to_cstring(key_arg);

//     entry = ram_cache_hash_lookup(ram_cache, key);
//     if (entry != NULL) {
//         // Free the memory associated with the entry
//         if (entry->key) pfree(entry->key);
//         if (entry->value) pfree(entry->value);
        
//         // Mark the entry as deleted
//         entry->status = 0;
//         entry->key = NULL;
//         entry->value = NULL;

//         PG_RETURN_BOOL(true);
//     }

//     PG_RETURN_BOOL(false);
// }

// // Clear the cache
// Datum ram_cache_clear(PG_FUNCTION_ARGS)
// {
//     if (ram_cache != NULL) {
//         ram_cache_hash_reset(ram_cache);
//         MemoryContextReset(ram_cache_context);
//     }

//     PG_RETURN_BOOL(true);
// }

// // Cleanup on unload
// void _PG_fini(void)
// {
//     if (ram_cache != NULL) {
//         ram_cache_hash_reset(ram_cache);
//         if (ram_cache_context != NULL) {
//             MemoryContextDelete(ram_cache_context);
//             ram_cache_context = NULL;
//             ram_cache = NULL;
//         }
//     }
// }





// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "utils/memutils.h"
// #include "utils/guc.h" 


// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// // Configurable parameters with default values
// static int ram_cache_max_entries = 10000;
// static int ram_cache_max_value_size = 4096;

// // Cache entry structure
// typedef struct {
//     char *key;           // Key identifier
//     char *value;         // Stored value
//     size_t value_length; // Length of the stored value
//     bool used;           // Entry usage flag
// } RamCacheEntry;

// // Global cache management structure
// typedef struct {
//     RamCacheEntry *entries;  // Array of cache entries
//     size_t max_size;         // Maximum number of entries
//     size_t current_entries;  // Current number of entries
//     MemoryContext cache_context; // Dedicated memory context
// } RamCacheManager;

// static RamCacheManager *ram_cache = NULL;

// // Plugin initialization function to register configuration parameters
// void _PG_init(void)
// {
//     // Similar configuration registration as in the file-based version
//     DefineCustomIntVariable(
//         "ram_cache.max_entries",
//         "Maximum number of entries in the RAM cache",
//         "Sets the total number of key-value pairs that can be stored in memory",
//         &ram_cache_max_entries,
//         10000,  // default
//         100,    // minimum
//         1000000,  // maximum
//         PGC_USERSET,
//         0,
//         NULL,
//         NULL,
//         NULL
//     );

//     DefineCustomIntVariable(
//         "ram_cache.max_value_size",
//         "Maximum size of a single cache entry value",
//         "Sets the maximum allowed size for a single cache entry value in bytes",
//         &ram_cache_max_value_size,
//         4096,   // default (4KB)
//         64,     // minimum
//         104857600,  // maximum (100MB)
//         PGC_USERSET,
//         0,
//         NULL,
//         NULL,
//         NULL
//     );
// }

// // Function prototypes
// PG_FUNCTION_INFO_V1(ram_cache_init);
// PG_FUNCTION_INFO_V1(ram_cache_set);
// PG_FUNCTION_INFO_V1(ram_cache_get);
// PG_FUNCTION_INFO_V1(ram_cache_clear);

// // Function to initialize RAM-based cache
// Datum ram_cache_init(PG_FUNCTION_ARGS)
// {
//     // Check if cache is already initialized
//     if (ram_cache != NULL) {
//         elog(NOTICE, "RAM Cache is already initialized");
//         PG_RETURN_BOOL(false);
//     }

//     // Create a dedicated memory context for the cache
//     MemoryContext old_context = MemoryContextSwitchTo(TopMemoryContext);
    
//     // Allocate cache management structure
//     ram_cache = palloc(sizeof(RamCacheManager));
    
//     // Initialize cache parameters
//     ram_cache->max_size = ram_cache_max_entries;
//     ram_cache->current_entries = 0;
    
//     // Allocate array for cache entries
//     ram_cache->entries = palloc0(
//         ram_cache_max_entries * sizeof(RamCacheEntry)
//     );
    
//     // Create a dedicated memory context for cache entries
//     ram_cache->cache_context = AllocSetContextCreate(
//         TopMemoryContext,
//         "RAMCacheContext",
//         ALLOCSET_DEFAULT_SIZES
//     );

//     // Switch back to original memory context
//     MemoryContextSwitchTo(old_context);

//     elog(NOTICE, "RAM Cache Initialized: Max Entries=%zu", 
//          ram_cache->max_size);

//     PG_RETURN_BOOL(true);
// }

// // Function to set a key-value pair
// Datum ram_cache_set(PG_FUNCTION_ARGS)
// {
//     // Validate cache initialization
//     if (ram_cache == NULL) {
//         elog(ERROR, "Cache not initialized. Call ram_cache_init first.");
//         PG_RETURN_BOOL(false);
//     }
    
//     // Get input arguments
//     text *key_arg = PG_GETARG_TEXT_P(0);
//     text *value_arg = PG_GETARG_TEXT_P(1);
    
//     // Convert to C strings
//     char *key_str = text_to_cstring(key_arg);
//     char *value_str = text_to_cstring(value_arg);
//     size_t value_len = strlen(value_str);

//     // Validate value size against configuration
//     if (value_len >= ram_cache_max_value_size) {
//         elog(NOTICE, "Value size %zu exceeds maximum configured size %d", 
//              value_len, ram_cache_max_value_size);
//         pfree(key_str);
//         pfree(value_str);
//         PG_RETURN_BOOL(false);
//     }

//     // Switch to cache memory context
//     MemoryContext old_context = MemoryContextSwitchTo(ram_cache->cache_context);

//     // Find first unused or matching entry
//     for (size_t i = 0; i < ram_cache->max_size; i++) {
//         if (!ram_cache->entries[i].used || 
//             strcmp(ram_cache->entries[i].key, key_str) == 0) {
            
//             // Free existing entry if updating
//             if (ram_cache->entries[i].key != NULL) {
//                 pfree(ram_cache->entries[i].key);
//             }
//             if (ram_cache->entries[i].value != NULL) {
//                 pfree(ram_cache->entries[i].value);
//             }

//             // Store new entry
//             ram_cache->entries[i].key = pstrdup(key_str);
//             ram_cache->entries[i].value = pstrdup(value_str);
//             ram_cache->entries[i].value_length = value_len;
//             ram_cache->entries[i].used = true;

//             // Update entry count if needed
//             if (!ram_cache->entries[i].used) {
//                 ram_cache->current_entries++;
//             }

//             // Switch back to original memory context
//             MemoryContextSwitchTo(old_context);

//             // Free temporary strings
//             pfree(key_str);
//             pfree(value_str);
            
//             PG_RETURN_BOOL(true);
//         }
//     }

//     // Switch back to original memory context
//     MemoryContextSwitchTo(old_context);

//     // No space found
//     pfree(key_str);
//     pfree(value_str);
//     PG_RETURN_BOOL(false);
// }

// // Function to retrieve a value
// Datum ram_cache_get(PG_FUNCTION_ARGS)
// {
//     // Validate cache initialization
//     if (ram_cache == NULL) {
//         elog(ERROR, "Cache not initialized. Call ram_cache_init first.");
//         PG_RETURN_NULL();
//     }
    
//     // Get input key
//     text *key_arg = PG_GETARG_TEXT_P(0);
//     char *key_str = text_to_cstring(key_arg);

//     // Search for matching entry
//     for (size_t i = 0; i < ram_cache->max_size; i++) {
//         if (ram_cache->entries[i].used && 
//             strcmp(ram_cache->entries[i].key, key_str) == 0) {
//             // Convert value to text
//             text *result = cstring_to_text_with_len(
//                 ram_cache->entries[i].value, 
//                 ram_cache->entries[i].value_length
//             );
//             pfree(key_str);
//             PG_RETURN_TEXT_P(result);
//         }
//     }

//     pfree(key_str);
//     PG_RETURN_NULL();
// }

// // Function to clear the cache
// Datum ram_cache_clear(PG_FUNCTION_ARGS)
// {
//     if (ram_cache != NULL) {
//         // Free all entries
//         for (size_t i = 0; i < ram_cache->max_size; i++) {
//             if (ram_cache->entries[i].used) {
//                 pfree(ram_cache->entries[i].key);
//                 pfree(ram_cache->entries[i].value);
//             }
//         }

//         // Free entries array and cache structure
//         pfree(ram_cache->entries);
        
//         // Destroy the memory context
//         MemoryContextDelete(ram_cache->cache_context);

//         // Free the cache management structure
//         pfree(ram_cache);
//         ram_cache = NULL;
//     }

//     PG_RETURN_BOOL(true);
// }

// // Unload hook to free memory when extension is unloaded
// void _PG_fini(void)
// {
//     ram_cache_clear(NULL);
// }













// #define _GNU_SOURCE
// #define _POSIX_C_SOURCE 200809L
// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "utils/memutils.h"
// #include "utils/guc.h"
// #include <limits.h>
// #include "utils/elog.h"
// #include <stdint.h>
// #include <sys/mman.h>
// #include <sys/stat.h>
// #include <fcntl.h>
// #include <unistd.h>
// #include <stdatomic.h>
// #include <time.h>

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// // Default Configuration Constants
// #define DEFAULT_CACHE_FILE_PATH "/dev/shm/pg_dynamic_cache"
// #define DEFAULT_MAX_CACHE_SIZE (1024 * 1024 * 128)  // 128MB default
// #define DEFAULT_MAX_ENTRIES 4096
// #define DEFAULT_MAX_KEY_LENGTH 256
// #define DEFAULT_MAX_VALUE_SIZE 262144  // 256KB
// #define CACHE_ENTRY_MAGIC 0xBADC0FFEE

// // Global Configuration Variables
// static char *cache_file_path = NULL;
// static int cache_max_entries = DEFAULT_MAX_ENTRIES;
// static int cache_max_key_length = DEFAULT_MAX_KEY_LENGTH;
// static int cache_max_value_size = DEFAULT_MAX_VALUE_SIZE;
// static int cache_max_size = DEFAULT_MAX_CACHE_SIZE;

// // Cache Entry Structure
// typedef struct {
//     uint64_t magic;            // Magic number for validation
//     uint64_t hash;             // Fast hash lookup
//     uint32_t key_length;       // Key length
//     uint32_t value_length;     // Value length
//     bool is_valid;             // Entry validity flag
//     time_t last_accessed;      // Timestamp for cache management
//     char data[];               // Flexible array for key and value
// } DynamicCacheEntry;

// // Memory-Mapped Cache Structure
// typedef struct {
//     int fd;                   // File descriptor
//     void *mapped_memory;      // Memory-mapped region
//     size_t mapped_size;       // Mapped memory size
//     size_t current_offset;    // Current writing offset
// } DynamicMMappedCache;

// // Static Global Cache Instance
// static DynamicMMappedCache *GlobalDynamicCache = NULL;

// // Ultra-Fast Hash Function
// static inline uint64_t ultrafast_hash(const char *key, size_t len) {
//     const uint64_t PRIME = 1099511628211ULL;
//     uint64_t hash = 14695981039346656037ULL;
    
//     for (size_t i = 0; i < len; i++) {
//         hash ^= (uint8_t)key[i];
//         hash *= PRIME;
//         hash ^= hash >> 33;
//     }
//     return hash;
// }

// // Find an Entry in the Cache
// static DynamicCacheEntry* find_cache_entry(const char *key, size_t key_len, uint64_t hash) {
//     char *current = (char*)GlobalDynamicCache->mapped_memory;
//     char *end = current + GlobalDynamicCache->current_offset;

//     while (current < end) {
//         DynamicCacheEntry *entry = (DynamicCacheEntry*)current;

//         // Validate entry
//         if (entry->magic != CACHE_ENTRY_MAGIC || !entry->is_valid) {
//             break;
//         }

//         // Compare hash, key length, and key contents
//         if (entry->hash == hash && 
//             entry->key_length == key_len && 
//             memcmp(entry->data, key, key_len) == 0) {
//             return entry;
//         }

//         // Move to next entry
//         current += sizeof(DynamicCacheEntry) + entry->key_length + entry->value_length;
//     }

//     return NULL;
// }

// // Dynamic Cache Initialization
// static void initialize_dynamic_cache(void) {
//     // Cleanup previous cache if exists
//     if (GlobalDynamicCache) {
//         if (GlobalDynamicCache->mapped_memory) {
//             munmap(GlobalDynamicCache->mapped_memory, GlobalDynamicCache->mapped_size);
//         }
//         if (GlobalDynamicCache->fd != -1) {
//             close(GlobalDynamicCache->fd);
//         }
//         pfree(GlobalDynamicCache);
//     }

//     // Allocate new cache structure
//     GlobalDynamicCache = palloc0(sizeof(DynamicMMappedCache));
//     GlobalDynamicCache->fd = -1;

//     // Open or create cache file
//     GlobalDynamicCache->fd = open(
//         cache_file_path ? cache_file_path : DEFAULT_CACHE_FILE_PATH, 
//         O_RDWR | O_CREAT | O_TRUNC, 
//         0600
//     );

//     if (GlobalDynamicCache->fd == -1) {
//         ereport(ERROR, (errmsg("Dynamic cache file creation failed")));
//     }

//     // Resize file to configured cache size
//     if (ftruncate(GlobalDynamicCache->fd, cache_max_size) == -1) {
//         close(GlobalDynamicCache->fd);
//         ereport(ERROR, (errmsg("Dynamic cache file resizing failed")));
//     }

//     // Memory map with performance flags
//     GlobalDynamicCache->mapped_memory = mmap(
//         NULL, 
//         cache_max_size, 
//         PROT_READ | PROT_WRITE, 
//         MAP_SHARED | MAP_POPULATE, 
//         GlobalDynamicCache->fd, 
//         0
//     );

//     if (GlobalDynamicCache->mapped_memory == MAP_FAILED) {
//         close(GlobalDynamicCache->fd);
//         ereport(ERROR, (errmsg("Dynamic cache mapping failed")));
//     }

//     // Initialize cache parameters
//     GlobalDynamicCache->mapped_size = cache_max_size;
//     GlobalDynamicCache->current_offset = 0;
// }

// // Configuration Initialization
// static void dynamic_cache_config_init(void) {
//     DefineCustomStringVariable(
//         "dynamic_cache.file_path",
//         "Path to the dynamic memory-mapped cache file",
//         "Sets the file path for dynamic cache",
//         &cache_file_path,
//         DEFAULT_CACHE_FILE_PATH,
//         PGC_USERSET,
//         0,
//         NULL,
//         NULL,
//         NULL
//     );

//     DefineCustomIntVariable(
//         "dynamic_cache.max_size",
//         "Maximum size of the dynamic cache in bytes",
//         "Sets the maximum cache size",
//         &cache_max_size,
//         DEFAULT_MAX_CACHE_SIZE,
//         1024 * 1024,  // Minimum 1MB
//         1024 * 1024 * 1024,  // Maximum 1GB
//         PGC_USERSET,
//         0,
//         NULL,
//         NULL,
//         NULL
//     );

//     DefineCustomIntVariable(
//         "dynamic_cache.max_entries",
//         "Maximum number of entries in dynamic cache",
//         "Sets the maximum number of cache entries",
//         &cache_max_entries,
//         DEFAULT_MAX_ENTRIES,
//         1,
//         INT_MAX,
//         PGC_USERSET,
//         0,
//         NULL,
//         NULL,
//         NULL
//     );

//     DefineCustomIntVariable(
//         "dynamic_cache.max_key_length",
//         "Maximum length of cache keys",
//         "Sets the maximum allowed key length",
//         &cache_max_key_length,
//         DEFAULT_MAX_KEY_LENGTH,
//         64,
//         8192,
//         PGC_USERSET,
//         0,
//         NULL,
//         NULL,
//         NULL
//     );

//     DefineCustomIntVariable(
//         "dynamic_cache.max_value_size",
//         "Maximum size of cached values in bytes",
//         "Sets the maximum allowed value size",
//         &cache_max_value_size,
//         DEFAULT_MAX_VALUE_SIZE,
//         4096,
//         1024 * 1024 * 100,
//         PGC_USERSET,
//         0,
//         NULL,
//         NULL,
//         NULL
//     );
// }

// // Cache Set Function
// PG_FUNCTION_INFO_V1(dynamic_cache_set);
// Datum dynamic_cache_set(PG_FUNCTION_ARGS) {
//     text *key_arg = PG_GETARG_TEXT_PP(0);
//     text *value_arg = PG_GETARG_TEXT_PP(1);
    
//     const char *key = VARDATA_ANY(key_arg);
//     const char *value = VARDATA_ANY(value_arg);
//     size_t key_len = VARSIZE_ANY_EXHDR(key_arg);
//     size_t value_len = VARSIZE_ANY_EXHDR(value_arg);

//     // Validation
//     if (key_len > cache_max_key_length || 
//         value_len > cache_max_value_size) {
//         PG_RETURN_BOOL(false);
//     }

//     // Compute hash
//     uint64_t hash = ultrafast_hash(key, key_len);

//     // Calculate total size needed
//     size_t total_size = sizeof(DynamicCacheEntry) + key_len + value_len;

//     // Check if we have enough space
//     if (GlobalDynamicCache->current_offset + total_size > GlobalDynamicCache->mapped_size) {
//         PG_RETURN_BOOL(false);
//     }

//     // Create new entry
//     DynamicCacheEntry *new_entry = 
//         (DynamicCacheEntry*)((char*)GlobalDynamicCache->mapped_memory + 
//                              GlobalDynamicCache->current_offset);

//     // Populate entry
//     new_entry->magic = CACHE_ENTRY_MAGIC;
//     new_entry->hash = hash;
//     new_entry->key_length = key_len;
//     new_entry->value_length = value_len;
//     new_entry->is_valid = true;
//     new_entry->last_accessed = time(NULL);

//     // Copy key and value
//     memcpy(new_entry->data, key, key_len);
//     memcpy(new_entry->data + key_len, value, value_len);

//     // Update current offset
//     GlobalDynamicCache->current_offset += total_size;

//     // Sync changes to disk
//     msync(GlobalDynamicCache->mapped_memory, GlobalDynamicCache->current_offset, MS_SYNC);

//     PG_RETURN_BOOL(true);
// }

// // Cache Get Function
// PG_FUNCTION_INFO_V1(dynamic_cache_get);
// Datum dynamic_cache_get(PG_FUNCTION_ARGS) {
//     text *key_arg = PG_GETARG_TEXT_PP(0);
    
//     const char *key = VARDATA_ANY(key_arg);
//     size_t key_len = VARSIZE_ANY_EXHDR(key_arg);

//     // Compute hash
//     uint64_t hash = ultrafast_hash(key, key_len);

//     // Find the entry
//     DynamicCacheEntry *entry = find_cache_entry(key, key_len, hash);
//     if (!entry) {
//         PG_RETURN_NULL();
//     }
    
//     // Convert to text
//     text *result = cstring_to_text_with_len(
//         entry->data + entry->key_length, 
//         entry->value_length
//     );

//     // Update last accessed time
//     entry->last_accessed = time(NULL);

//     PG_RETURN_TEXT_P(result);
// }

// // New function for configuration change hook registration
// PG_FUNCTION_INFO_V1(register_dynamic_cache_config_hook);
// Datum register_dynamic_cache_config_hook(PG_FUNCTION_ARGS) {
//     // This function will be called to register the configuration change hook
//     // It uses pgtle.register_feature to register the hook
    
//     // In this example, we'll use a hypothetical hook name 'dynamic_cache_config'
//     const char *hook_name = "dynamic_cache_config";
    
//     // Call the registration function
//     // Note: In a real implementation, you'd need to ensure this function exists
//     // and handles the actual hook registration
//     SPI_connect();
    
//     // Construct and execute the registration query
//     StringInfoData query;
//     initStringInfo(&query);
//     appendStringInfo(&query, 
//         "SELECT pgtle.register_feature('%s', '%s')", 
//         "dynamic_cache_config_change_hook", 
//         hook_name
//     );
    
//     // Execute the registration query
//     int ret = SPI_exec(query.data, 0);
    
//     // Clean up
//     SPI_finish();
    
//     // Return true if registration was successful
//     PG_RETURN_BOOL(ret >= 0);
// }

// // Extension Initialization
// void *pg_init(void) {
//     // Register configuration initialization
//     dynamic_cache_config_init();
    
//     // Initialize cache
//     initialize_dynamic_cache();
    
//     return NULL;
// }


// // Cleanup Function
// void *pg_fini(void) {
//     if (GlobalDynamicCache) {
//         if (GlobalDynamicCache->mapped_memory) {
//             munmap(GlobalDynamicCache->mapped_memory, GlobalDynamicCache->mapped_size);
//         }
        
//         if (GlobalDynamicCache->fd != -1) {
//             close(GlobalDynamicCache->fd);
//             unlink(cache_file_path ? cache_file_path : DEFAULT_CACHE_FILE_PATH);
//         }
        
//         pfree(GlobalDynamicCache);
//         GlobalDynamicCache = NULL;
//     }
//     return NULL;
// }

// // Optional Cache Clear Function
// PG_FUNCTION_INFO_V1(dynamic_cache_clear);
// Datum dynamic_cache_clear(PG_FUNCTION_ARGS) {
//     // Reset the current offset to effectively clear the cache
//     if (GlobalDynamicCache) {
//         GlobalDynamicCache->current_offset = 0;
//         memset(GlobalDynamicCache->mapped_memory, 0, GlobalDynamicCache->mapped_size);
//     }
    
//     PG_RETURN_BOOL(true);
// }































// #include <xxhash.h



// #define _GNU_SOURCE
// #define _POSIX_C_SOURCE 200809L
// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "utils/memutils.h"
// #include "utils/guc.h"
// #include <limits.h>
// #include "utils/elog.h"
// #include <stdint.h>
// #include <zlib.h>  // For compression

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// // Enhanced configuration constants
// #define DEFAULT_MAX_CACHE_ENTRIES 4096
// #define DEFAULT_MAX_KEY_LENGTH 512
// #define DEFAULT_MAX_VALUE_SIZE 524288
// #define CACHE_LOAD_FACTOR 0.75
// #define INITIAL_CAPACITY 1024
// #define COMPRESSION_THRESHOLD 4096  // Compress values larger than 4KB
// #define MAX_COMPRESSION_BUFFER 1024 * 1024  // 1MB max compressed buffer

// // Improved hash function (similar to previous implementation)
// static inline uint64_t optimized_hash(const char *key, size_t len) {
//     const uint64_t PRIME64_1 = 11400714785074694791ULL;
//     const uint64_t PRIME64_2 = 14029467366897019727ULL;
//     const uint64_t PRIME64_3 = 1609587929392839161ULL;

//     uint64_t hash = PRIME64_1;
//     for (size_t i = 0; i < len; i++) {
//         hash ^= (uint8_t)key[i];
//         hash *= PRIME64_2;
//         hash = ((hash << 31) | (hash >> 33)) * PRIME64_3;
//     }

//     hash ^= hash >> 33;
//     hash *= PRIME64_2;
//     hash ^= hash >> 29;
//     return hash;
// }

// // Enhanced cache entry with compression support
// typedef struct {
//     uint64_t hash;         // 64-bit hash
//     uint32_t key_length;   // Key length
//     uint32_t value_length; // Original value length
//     uint32_t compressed_length;  // Compressed length (0 if not compressed)
//     bool is_compressed;    // Compression flag
//     char data[];           // Flexible array for key and compressed/uncompressed data
// } OptimizedCacheEntry;

// // Enhanced cache structure with more robust management
// typedef struct {
//     OptimizedCacheEntry **entries;   // Dynamic entry array
//     int capacity;                    // Total slots
//     int size;                        // Current number of entries
//     int max_entries;                 // Configurable max entries
//     MemoryContext cache_context;     // Dedicated memory context
// } DynamicCache;

// // Global cache instance
// static DynamicCache *UltraCache = NULL;

// // Configuration variables
// static int cache_max_entries = DEFAULT_MAX_CACHE_ENTRIES;
// static int cache_max_key_length = DEFAULT_MAX_KEY_LENGTH;
// static int cache_max_value_size = DEFAULT_MAX_VALUE_SIZE;

// // Compression utility functions
// static bool compress_value(const char *value, size_t value_len, 
//                            char **compressed_data, size_t *compressed_len) {
//     if (value_len < COMPRESSION_THRESHOLD) {
//         return false;
//     }

//     // Allocate compression buffer
//     uLong dest_len = compressBound(value_len);
//     *compressed_data = palloc(dest_len);

//     // Compress the data
//     int result = compress2((Bytef *)*compressed_data, &dest_len, 
//                            (const Bytef *)value, value_len, 
//                            Z_BEST_SPEED);
    
//     if (result != Z_OK || dest_len > MAX_COMPRESSION_BUFFER) {
//         pfree(*compressed_data);
//         return false;
//     }

//     *compressed_len = dest_len;
//     return true;
// }

// static bool decompress_value(const char *compressed_data, size_t compressed_len, 
//                              size_t original_len, char **decompressed_data) {
//     *decompressed_data = palloc(original_len);
//     uLong dest_len = original_len;

//     int result = uncompress((Bytef *)*decompressed_data, &dest_len, 
//                             (const Bytef *)compressed_data, compressed_len);
    
//     if (result != Z_OK) {
//         pfree(*decompressed_data);
//         return false;
//     }

//     return true;
// }
// // Specialized cache initialization with dynamic allocation
// static void ultra_cache_dynamic_init(void) {
//     MemoryContext old_context = CurrentMemoryContext;
    
//     UltraCache = MemoryContextAllocZero(
//         TopMemoryContext, 
//         sizeof(DynamicCache)
//     );

//     UltraCache->cache_context = AllocSetContextCreate(
//         TopMemoryContext,
//         "UltraCache",
//         ALLOCSET_DEFAULT_SIZES
//     );

//     MemoryContextSwitchTo(UltraCache->cache_context);

//     // Initial capacity with room for growth
//     UltraCache->entries = palloc0(
//         INITIAL_CAPACITY * sizeof(OptimizedCacheEntry *)
//     );
//     UltraCache->capacity = INITIAL_CAPACITY;
//     UltraCache->max_entries = cache_max_entries;

//     MemoryContextSwitchTo(old_context);
// }

// // Resize cache function (similar to previous implementation)
// static bool resize_cache(DynamicCache *cache, int new_capacity) {
//     MemoryContext old_context = MemoryContextSwitchTo(cache->cache_context);

//     OptimizedCacheEntry **new_entries = palloc0(
//         new_capacity * sizeof(OptimizedCacheEntry *)
//     );

//     // Rehash existing entries with quadratic probing
//     for (int i = 0; i < cache->capacity; i++) {
//         if (cache->entries[i]) {
//             uint64_t hash = cache->entries[i]->hash;
//             uint32_t index = hash % new_capacity;

//             int j, attempt;
//             for (attempt = 0; attempt < new_capacity; attempt++) {
//                 j = (index + attempt * attempt) % new_capacity;
//                 if (!new_entries[j]) {
//                     new_entries[j] = cache->entries[i];
//                     break;
//                 }
//             }

//             if (attempt == new_capacity) {
//                 MemoryContextSwitchTo(old_context);
//                 return false;
//             }
//         }
//     }

//     pfree(cache->entries);
//     cache->entries = new_entries;
//     cache->capacity = new_capacity;

//     MemoryContextSwitchTo(old_context);
//     return true;
// }

// // Optimized set function with compression support
// PG_FUNCTION_INFO_V1(ultra_fast_cache_set);
// Datum ultra_fast_cache_set(PG_FUNCTION_ARGS) {
//     text *key_arg = PG_GETARG_TEXT_PP(0);
//     text *value_arg = PG_GETARG_TEXT_PP(1);
    
//     const char *key = VARDATA_ANY(key_arg);
//     const char *value = VARDATA_ANY(value_arg);
//     size_t key_len = VARSIZE_ANY_EXHDR(key_arg);
//     size_t value_len = VARSIZE_ANY_EXHDR(value_arg);

//     // Validation
//     if (key_len >= cache_max_key_length || 
//         value_len > cache_max_value_size) {
//         ereport(ERROR, (errmsg("Key or value size exceeds limits")));
//     }

//     // Check if resize needed
//     if (UltraCache->size >= (int)(UltraCache->capacity * CACHE_LOAD_FACTOR)) {
//         int new_capacity = UltraCache->capacity * 2;
//         if (!resize_cache(UltraCache, new_capacity)) {
//             ereport(ERROR, (errmsg("Unable to resize cache")));
//         }
//     }

//     MemoryContext old_context = MemoryContextSwitchTo(UltraCache->cache_context);

//     // Compute hash
//     uint64_t hash = optimized_hash(key, key_len);
//     uint32_t index = hash % UltraCache->capacity;

//     // Compression preparation
//     char *compressed_data = NULL;
//     size_t compressed_len = 0;
//     bool is_compressed = compress_value(value, value_len, &compressed_data, &compressed_len);

//     // Quadratic probing
//     int attempt;
//     for (attempt = 0; attempt < UltraCache->capacity; attempt++) {
//         int current_index = (index + attempt * attempt) % UltraCache->capacity;
//         OptimizedCacheEntry *entry = UltraCache->entries[current_index];

//         // Slot available or key match
//         if (!entry || 
//             (entry->hash == hash && 
//              entry->key_length == key_len && 
//              memcmp(entry->data, key, key_len) == 0)) {
            
//             // Free existing entry if updating
//             if (entry) pfree(entry);

//             // Determine total size and data to store
//             size_t total_size = sizeof(OptimizedCacheEntry) + key_len + 
//                                 (is_compressed ? compressed_len : value_len);
//             entry = palloc(total_size);
            
//             entry->hash = hash;
//             entry->key_length = key_len;
//             entry->value_length = value_len;
//             entry->is_compressed = is_compressed;
            
//             // Copy key
//             memcpy(entry->data, key, key_len);
            
//             // Copy value (compressed or uncompressed)
//             if (is_compressed) {
//                 entry->compressed_length = compressed_len;
//                 memcpy(entry->data + key_len, compressed_data, compressed_len);
//                 pfree(compressed_data);
//             } else {
//                 entry->compressed_length = 0;
//                 memcpy(entry->data + key_len, value, value_len);
//             }

//             UltraCache->entries[current_index] = entry;

//             // Update entry count
//             if (!entry) UltraCache->size++;

//             MemoryContextSwitchTo(old_context);
//             PG_RETURN_BOOL(true);
//         }
//     }

//     MemoryContextSwitchTo(old_context);
//     ereport(ERROR, (errmsg("Cache full, unable to insert")));
// }

// // Optimized retrieval function with decompression
// PG_FUNCTION_INFO_V1(ultra_fast_cache_get);
// Datum ultra_fast_cache_get(PG_FUNCTION_ARGS) {
//     text *key_arg = PG_GETARG_TEXT_PP(0);
    
//     const char *key = VARDATA_ANY(key_arg);
//     size_t key_len = VARSIZE_ANY_EXHDR(key_arg);

//     // Compute hash
//     uint64_t hash = optimized_hash(key, key_len);
//     uint32_t index = hash % UltraCache->capacity;

//     // Quadratic probing search
//     int attempt;
//     for (attempt = 0; attempt < UltraCache->capacity; attempt++) {
//         int current_index = (index + attempt * attempt) % UltraCache->capacity;
//         OptimizedCacheEntry *entry = UltraCache->entries[current_index];

//         // Early exit conditions
//         if (!entry) break;

//         // Fast comparison with precomputed hash
//         if (entry->hash == hash && 
//             entry->key_length == key_len && 
//             memcmp(entry->data, key, key_len) == 0) {
            
//             // Determine how to retrieve value
//             char *value_to_return;
//             size_t return_len;

//             if (entry->is_compressed) {
//                 // Decompress if needed
//                 char *decompressed_data;
//                 if (!decompress_value(
//                     entry->data + key_len, 
//                     entry->compressed_length, 
//                     entry->value_length, 
//                     &decompressed_data
//                 )) {
//                     // Decompression failed
//                     PG_RETURN_NULL();
//                 }
//                 value_to_return = decompressed_data;
//                 return_len = entry->value_length;
//             } else {
//                 // Direct retrieval
//                 value_to_return = entry->data + key_len;
//                 return_len = entry->value_length;
//             }
            
//             // Convert to text
//             text *result = cstring_to_text_with_len(
//                 value_to_return, 
//                 return_len
//             );

//             // Free decompressed data if needed
//             if (entry->is_compressed) {
//                 pfree(value_to_return);
//             }

//             PG_RETURN_TEXT_P(result);
//         }
//     }

//     PG_RETURN_NULL();
// }

// // Configuration initialization (similar to previous implementation)
// static void ultra_cache_config_init(void) {
//     DefineCustomIntVariable(
//         "ultra_cache.max_entries",
//         "Maximum number of entries in UltraCache",
//         "Sets the maximum number of cache entries",
//         &cache_max_entries,
//         DEFAULT_MAX_CACHE_ENTRIES,
//         1024,
//         INT_MAX,
//         PGC_USERSET,
//         0,
//         NULL,
//         NULL,
//         NULL
//     );

//     DefineCustomIntVariable(
//         "ultra_cache.max_key_length",
//         "Maximum length of cache keys",
//         "Sets the maximum allowed key length",
//         &cache_max_key_length,
//         DEFAULT_MAX_KEY_LENGTH,
//         64,
//         8192,
//         PGC_USERSET,
//         0,
//         NULL,
//         NULL,
//         NULL
//     );

//     DefineCustomIntVariable(
//         "ultra_cache.max_value_size",
//         "Maximum size of cached values in bytes",
//         "Sets the maximum allowed value size",
//         &cache_max_value_size,
//         DEFAULT_MAX_VALUE_SIZE,
//         4096,
//         1024 * 1024 * 100,
//         PGC_USERSET,
//         0,
//         NULL,
//         NULL,
//         NULL
//     );
// }

// // Extension initialization
// void _PG_init(void) {
//     ultra_cache_config_init();
//     ultra_cache_dynamic_init();
// }

// // Enhanced cleanup
// void _PG_fini(void) {
//     if (UltraCache && UltraCache->cache_context) {
//         for (int i = 0; i < UltraCache->capacity; i++) {
//             if (UltraCache->entries[i]) {
//                 pfree(UltraCache->entries[i]);
//             }
//         }
//         MemoryContextDelete(UltraCache->cache_context);
//         UltraCache = NULL;
//     }
// }


// Optimized set function with dynamic resizing


// // Hyper-Optimized Cache Configuration
// #define MAX_CACHE_ENTRIES 32        // Increased for better hash distribution
// #define MAX_KEY_LENGTH 48            // Reduced to optimize memory
// #define MAX_FILENAME_LEN 192         // More unique filename space
// #define MMAP_FILE_PREFIX "/dev/shm/pg_ultra_cache_"  // Use RAM disk for faster I/O
// #define CACHE_ENTRY_LIFETIME 1800    // Reduced to 30 minutes

// // Advanced Cache Entry Structure with Memory Alignment
// typedef struct __attribute__((packed)) {
//     char key[MAX_KEY_LENGTH];        // Compact key storage
//     char *value_path;                // Pointer to memory-mapped file path
//     size_t value_size;               // Actual value size
//     int mmap_fd;                     // File descriptor for memory mapping
//     void *mmap_ptr;                  // Pointer to memory-mapped data
//     uint32_t hash_key;               // Pre-computed hash for faster lookup
//     uint64_t creation_time;          // Nanosecond timestamp for precision
//     bool is_active;                  // Compact flag for entry status
// } UltraMicroCacheEntry;

// // Optimized Global Cache Structure
// static struct {
//     UltraMicroCacheEntry *entries;   // Dynamic entry array
//     MemoryContext context;           // Dedicated memory context
//     volatile int access_counter;     // Thread-safe access counter
//     volatile bool is_initialized;    // Atomic initialization flag
// } UltraCache = {0};

// // High-Performance Hash Function (xxHash-inspired)
// static inline uint32_t 
// ultra_optimized_hash(const char *key, size_t len) 
// {
//     uint32_t hash = 2166136261U;  // FNV-1a initial basis
//     for (size_t i = 0; i < len; i++) {
//         hash ^= key[i];
//         hash *= 16777619U;  // FNV prime
//     }
//     return hash % MAX_CACHE_ENTRIES;
// }

// // Nanosecond Precise Timestamp
// static inline uint64_t 
// get_precise_timestamp(void) 
// {
//     struct timespec ts;
//     // Use CLOCK_MONOTONIC if CLOCK_MONOTONIC_RAW is not available
//     clock_gettime(CLOCK_MONOTONIC, &ts);
//     return (ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
// }

// // Ultra-Fast Filename Generation
// static void
// generate_optimized_filename(char *filename, size_t max_len) 
// {
//     // Use atomic increment for thread safety
//     int unique_id = __sync_fetch_and_add(&UltraCache.access_counter, 1);
//     uint64_t timestamp = get_precise_timestamp();
    
//     snprintf(filename, max_len, "%s%d_%d_%lu", 
//              MMAP_FILE_PREFIX, 
//              MyProcPid, 
//              unique_id,
//              timestamp);
// }
// // Efficient Cache Entry Cleanup
// static void
// cleanup_cache_entry(UltraMicroCacheEntry *entry) 
// {
//     if (!entry->is_active) return;

//     // Quick atomic memory unmapping
//     if (entry->mmap_ptr && entry->value_size) {
//         munmap(entry->mmap_ptr, entry->value_size);
//     }

//     // Rapid file descriptor management
//     if (entry->mmap_fd != -1) {
//         close(entry->mmap_fd);
//     }

//     // Quick file removal
//     if (entry->value_path) {
//         unlink(entry->value_path);
//     }

//     // Zero-memory reset
//     memset(entry, 0, sizeof(UltraMicroCacheEntry));
// }

// // Highly Optimized Cache Initialization
// PG_FUNCTION_INFO_V1(ultra_micro_cache_init);
// Datum ultra_micro_cache_init(PG_FUNCTION_ARGS)
// {
//     MemoryContext old_context;

//     // Atomic check to prevent multiple initializations
//     if (__sync_bool_compare_and_swap(&UltraCache.is_initialized, false, true)) {
//         // Create memory context with minimal overhead
//         UltraCache.context = AllocSetContextCreate(
//             TopMemoryContext,
//             "UltraFastCache",
//             ALLOCSET_SMALL_SIZES
//         );

//         // Allocate entries in one go
//         old_context = MemoryContextSwitchTo(UltraCache.context);
//         UltraCache.entries = palloc0(sizeof(UltraMicroCacheEntry) * MAX_CACHE_ENTRIES);
//         MemoryContextSwitchTo(old_context);

//         // Pre-initialize entry states
//         for (int i = 0; i < MAX_CACHE_ENTRIES; i++) {
//             UltraCache.entries[i].mmap_fd = -1;
//         }
//     }

//     PG_RETURN_BOOL(true);
// }



// // Optimized Cache Set Function
// PG_FUNCTION_INFO_V1(ultra_micro_cache_set);
// Datum ultra_micro_cache_set(PG_FUNCTION_ARGS)
// {
//     text *key_arg, *value_arg;
//     char *key;
//     char *value;
//     int32 key_len;
//     int32 value_len;
//     uint32_t hash_index;
//     char filename[MAX_FILENAME_LEN];
//     int mmap_fd = -1;
//     void *mmap_ptr = MAP_FAILED;
//     uint32_t pre_computed_hash;
//     UltraMicroCacheEntry *cache_entry;
    
//     // Fast null checks
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
//         PG_RETURN_BOOL(false);

//     // Ensure cache is initialized
//     if (!UltraCache.is_initialized) 
//         ultra_micro_cache_init(fcinfo);

//     // Efficient argument processing
//     key_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
//     value_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(1));

//     key = VARDATA_ANY(key_arg);
//     value = VARDATA_ANY(value_arg);
//     key_len = VARSIZE_ANY_EXHDR(key_arg);
//     value_len = VARSIZE_ANY_EXHDR(value_arg);

//     // Rapid input validation
//     if (key_len >= MAX_KEY_LENGTH) {
//         pfree(key_arg);
//         pfree(value_arg);
//         PG_RETURN_BOOL(false);
//     }

//     // Compute hash index
//     hash_index = ultra_optimized_hash(key, key_len);
//     pre_computed_hash = ultra_optimized_hash(key, key_len);

//     // Quick entry cleanup
//     if (UltraCache.entries[hash_index].is_active) {
//         cleanup_cache_entry(&UltraCache.entries[hash_index]);
//     }

//     // Generate unique filename using RAM disk
//     generate_optimized_filename(filename, sizeof(filename));
//     mmap_fd = open(filename, O_RDWR | O_CREAT | O_TRUNC | PG_O_DIRECT, 0600);
    
//     if (mmap_fd == -1) {
//         pfree(key_arg);
//         pfree(value_arg);
//         PG_RETURN_BOOL(false);
//     }

//     // Efficient memory mapping
//     if (ftruncate(mmap_fd, value_len) == -1 ||
//     (mmap_ptr = mmap(NULL, value_len, PROT_WRITE, 
//                      MAP_SHARED | MAP_POPULATE, 
//                      mmap_fd, 0)) == MAP_FAILED) {
//         close(mmap_fd);
//         unlink(filename);
//         pfree(key_arg);
//         pfree(value_arg);
//         PG_RETURN_BOOL(false);
//     }

//     // Ultra-fast memory copy with prefetching
//     __builtin_prefetch(mmap_ptr, 1, 1);
//     memcpy(mmap_ptr, value, value_len);
//     msync(mmap_ptr, value_len, MS_ASYNC);

//     // Compact entry storage
//     cache_entry = &UltraCache.entries[hash_index];
//     memset(cache_entry->key, 0, MAX_KEY_LENGTH);
//     memcpy(cache_entry->key, key, key_len);
    
//     cache_entry->value_path = pstrdup(filename);
//     cache_entry->value_size = value_len;
//     cache_entry->mmap_fd = mmap_fd;
//     cache_entry->mmap_ptr = mmap_ptr;
//     cache_entry->is_active = true;
//     cache_entry->creation_time = get_precise_timestamp();
//     cache_entry->hash_key = pre_computed_hash;

//     pfree(key_arg);
//     pfree(value_arg);

//     PG_RETURN_BOOL(true);
// }

// PG_FUNCTION_INFO_V1(ultra_micro_cache_get);
// Datum ultra_micro_cache_get(PG_FUNCTION_ARGS)
// {
//     text *key_arg;
//     char *key;
//     int32 key_len;
//     uint32_t hash_index;
//     text *result_text;
//     UltraMicroCacheEntry *cache_entry;

//     if (PG_ARGISNULL(0) || !UltraCache.is_initialized)
//         PG_RETURN_NULL();

//     key_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
//     key = VARDATA_ANY(key_arg);
//     key_len = VARSIZE_ANY_EXHDR(key_arg);

//     // Simplified hash computation
//     hash_index = ultra_optimized_hash(key, key_len);
//     cache_entry = &UltraCache.entries[hash_index];

//     // Ultra-fast, minimal comparison
//     if (cache_entry->is_active &&
//         strlen(cache_entry->key) == key_len &&
//         memcmp(cache_entry->key, key, key_len) == 0) {
        
//         result_text = cstring_to_text_with_len(
//             cache_entry->mmap_ptr,
//             cache_entry->value_size
//         );

//         elog(INFO, "Value size: %zu bytes", cache_entry->value_size);
//         pfree(key_arg);
//         PG_RETURN_TEXT_P(result_text);
//     }

//     pfree(key_arg);
//     PG_RETURN_NULL();
// }

// // Efficient Cleanup Function
// PG_FUNCTION_INFO_V1(ultra_micro_cache_cleanup);
// Datum ultra_micro_cache_cleanup(PG_FUNCTION_ARGS)
// {
//     uint64_t current_time = get_precise_timestamp();

//     if (!UltraCache.is_initialized)
//         PG_RETURN_BOOL(false);

//     for (int i = 0; i < MAX_CACHE_ENTRIES; i++) {
//         if (UltraCache.entries[i].is_active &&
//             (current_time - UltraCache.entries[i].creation_time) > 
//             (CACHE_ENTRY_LIFETIME * 1000000000ULL)) {
//             cleanup_cache_entry(&UltraCache.entries[i]);
//         }
//     }

//     PG_RETURN_BOOL(true);
// }

// // Optimized Shutdown Function
// PG_FUNCTION_INFO_V1(ultra_micro_cache_shutdown);
// Datum ultra_micro_cache_shutdown(PG_FUNCTION_ARGS)
// {
//     if (UltraCache.is_initialized) {
//         for (int i = 0; i < MAX_CACHE_ENTRIES; i++) {
//             cleanup_cache_entry(&UltraCache.entries[i]);
//         }

//         pfree(UltraCache.entries);
//         MemoryContextDelete(UltraCache.context);
        
//         // Reset global structure
//         memset(&UltraCache, 0, sizeof(UltraCache));
//     }

//     PG_RETURN_BOOL(true);
// }














// #define _GNU_SOURCE
// #define _POSIX_C_SOURCE 200809L


// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "utils/palloc.h"
// #include "utils/memutils.h"
// #include "storage/fd.h"
// #include "miscadmin.h"
// #include "port/atomics.h"

// #include <sys/mman.h>
// #include <fcntl.h>
// #include <unistd.h>
// #include <time.h>
// #include <string.h>
// #include <sys/stat.h>
// #include <errno.h>
// #include <stdint.h>
// #include "utils/geo_decls.h"
// #include "varatt.h"


// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif


// // Enhanced Cache Configuration
// #define MAX_CACHE_ENTRIES 1024
// #define MAX_KEY_LENGTH 256
// #define MAX_FILENAME_LEN 512
// #define CACHE_ENTRY_LIFETIME 3600  // 1 hour in seconds
// #define MAX_DIRECT_CACHE_SIZE (4LL * 1024 * 1024)  // 4 MB direct cache
// #define MAX_MEDIUM_CACHE_SIZE (64LL * 1024 * 1024)  // 64 MB medium cache
// #define MAX_STREAM_BUFFER_SIZE (64 * 1024)  // 64 KB streaming buffer

// // Logging Macro
// #define CACHE_LOG(level, ...) 
//     do { 
//         elog(level, "UltraCache: " __VA_ARGS__); 
//     } while (0)

// // Storage Method Enum
// typedef enum {
//     STORAGE_DIRECT_MEMORY,
//     STORAGE_MMAP_FILE,
//     STORAGE_LARGE_FILE
// } storage_method_t;

// // Enhanced Cache Entry Structure
// typedef struct __attribute__((packed)) {
//     char key[MAX_KEY_LENGTH];
//     union {
//         struct {
//             void *ptr;           // Direct pointer to data
//             size_t size;         // Actual data size
//         } direct;
//         struct {
//             int fd;               // File descriptor
//             off_t offset;         // Offset in file
//             size_t size;          // Data size
//             void *mmap_ptr;       // Memory-mapped pointer
//         } mapped;
//     } data;
//     uint64_t creation_time;
//     uint32_t hash_key;
//     storage_method_t storage_method;
//     bool is_active;
// } OptimizedCacheEntry;


// static bool initialize_cache_directory(void);
// static void cleanup_cache_entry(OptimizedCacheEntry *entry);
// static bool generate_optimized_filename(char *filename, size_t max_len);
// static inline uint32_t ultra_optimized_hash(const char *key, size_t len);
// static inline uint64_t get_precise_timestamp(void);

// Datum ultra_micro_cache_set(PG_FUNCTION_ARGS);
// Datum ultra_micro_cache_delete(PG_FUNCTION_ARGS);

// static struct {
//     OptimizedCacheEntry *entries;
//     MemoryContext context;
//     volatile bool is_initialized;
//     char base_cache_dir[MAX_FILENAME_LEN];
//     pg_atomic_uint64 total_cache_size;
//     volatile int access_counter;
// } UltraCache = {0};

// // Quick Access Buffer for Large Files
// static struct {
//     char buffer[MAX_STREAM_BUFFER_SIZE];
//     pg_atomic_flag lock;
// } QuickAccessBuffer;

// // Efficient Hash Function (FNV-1a variant)
// static inline uint32_t 
// ultra_optimized_hash(const char *key, size_t len) 
// {
//     uint32_t hash = 2166136261U;
//     for (size_t i = 0; i < len; i++) {
//         hash ^= (unsigned char)key[i];
//         hash *= 16777619U;
//     }
//     return hash % MAX_CACHE_ENTRIES;
// }

// // Precise Timestamp Function
// static inline uint64_t 
// get_precise_timestamp(void) 
// {
//     struct timespec ts;
//     clock_gettime(CLOCK_MONOTONIC, &ts);
//     return (ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
// }

// // Generate Unique Filename
// static bool
// generate_optimized_filename(char *filename, size_t max_len) 
// {
//     int unique_id = __sync_fetch_and_add(&UltraCache.access_counter, 1);
//     uint64_t timestamp = get_precise_timestamp();
    
//     int result = snprintf(filename, max_len, 
//         "/tmp/pg_ultra_cache_%d_%d_%lu.cache", 
//         MyProcPid, unique_id, timestamp);
    
//     return result > 0 && (size_t)result < max_len;
// }

// // Initialize Base Cache Directory
// static bool
// initialize_cache_directory(void) 
// {
//     int result = snprintf(UltraCache.base_cache_dir, 
//                           sizeof(UltraCache.base_cache_dir), 
//                           "/tmp/pg_ultra_cache_%d", 
//                           MyProcPid);
    
//     if (result < 0 || (size_t)result >= sizeof(UltraCache.base_cache_dir)) {
//         CACHE_LOG(ERROR, "Failed to generate cache directory path");
//         return false;
//     }
    
//     if (mkdir(UltraCache.base_cache_dir, 0700) != 0 && errno != EEXIST) {
//         CACHE_LOG(WARNING, "Could not create ultra cache directory");
//         return false;
//     }
    
//     return true;
// }

// // Cleanup Cache Entry
// static void
// cleanup_cache_entry(OptimizedCacheEntry *entry) 
// {
//     if (!entry->is_active) return;

//     switch (entry->storage_method) {
//         case STORAGE_DIRECT_MEMORY:
//             if (entry->data.direct.ptr) {
//                 pfree(entry->data.direct.ptr);
//             }
//             break;
        
//         case STORAGE_MMAP_FILE:
//             if (entry->data.mapped.mmap_ptr && 
//                 entry->data.mapped.mmap_ptr != MAP_FAILED) {
//                 munmap(entry->data.mapped.mmap_ptr, entry->data.mapped.size);
//             }
//             if (entry->data.mapped.fd != -1) {
//                 close(entry->data.mapped.fd);
//             }
//             break;
        
//         case STORAGE_LARGE_FILE:
//             if (entry->data.mapped.fd != -1) {
//                 close(entry->data.mapped.fd);
//             }
//             break;
//     }

//     // Reset entry
//     memset(entry, 0, sizeof(OptimizedCacheEntry));
//     entry->data.mapped.fd = -1;
// }

// // Cache Initialization Function
// PG_FUNCTION_INFO_V1(ultra_micro_cache_init);
// Datum ultra_micro_cache_init(PG_FUNCTION_ARGS)
// {
//     MemoryContext old_context;

//     // Use atomic compare and swap for thread-safe initialization
//     if (__sync_bool_compare_and_swap(&UltraCache.is_initialized, false, true)) {
//         // Initialize base cache directory
//         if (!initialize_cache_directory()) {
//             UltraCache.is_initialized = false;
//             PG_RETURN_BOOL(false);
//         }

//         // Create memory context
//         UltraCache.context = AllocSetContextCreate(
//             TopMemoryContext,
//             "UltraLargeCache",
//             ALLOCSET_SMALL_SIZES
//         );

//         // Switch to custom memory context
//         old_context = MemoryContextSwitchTo(UltraCache.context);
        
//         // Allocate entries with zero initialization
//         UltraCache.entries = palloc0(sizeof(OptimizedCacheEntry) * MAX_CACHE_ENTRIES);
        
//         // Restore previous memory context
//         MemoryContextSwitchTo(old_context);

//         // Pre-initialize entry states
//         for (int i = 0; i < MAX_CACHE_ENTRIES; i++) {
//             UltraCache.entries[i].data.mapped.fd = -1;
//         }

//         // Initialize atomic counters
//         pg_atomic_init_u64(&UltraCache.total_cache_size, 0);
//     }

//     PG_RETURN_BOOL(true);
// }

// PG_FUNCTION_INFO_V1(ultra_micro_cache_set);
// Datum ultra_micro_cache_set(PG_FUNCTION_ARGS)
// {
//     text *key_arg, *value_arg;
//     char *key, *value;
//     size_t key_len, value_len;
//     uint32_t hash_index;
//     char filename[MAX_FILENAME_LEN];
//     OptimizedCacheEntry *cache_entry;
//     int write_result;
    
//     // Validate input arguments
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
//         PG_RETURN_BOOL(false);

//     // Ensure cache is initialized
//     if (!UltraCache.is_initialized) 
//         ultra_micro_cache_init(fcinfo);

//     // Process arguments with proper detoasting
//     key_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
//     value_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(1));

//     // Use explicit casting and VARDATA_ANY
//     key = (char *)VARDATA_ANY(key_arg);
//     value = (char *)VARDATA_ANY(value_arg);
//     key_len = VARSIZE_ANY_EXHDR(key_arg);
//     value_len = VARSIZE_ANY_EXHDR(value_arg);

//     // Validate input sizes
//     if (key_len >= MAX_KEY_LENGTH || value_len > MAX_MEDIUM_CACHE_SIZE) {
//         pfree(key_arg);
//         pfree(value_arg);
//         CACHE_LOG(NOTICE, "Key or value size exceeds limits");
//         PG_RETURN_BOOL(false);
//     }

//     // Compute hash index
//     hash_index = ultra_optimized_hash(key, key_len);
//     cache_entry = &UltraCache.entries[hash_index];

//     // Cleanup existing entry
//     if (cache_entry->is_active) {
//         cleanup_cache_entry(cache_entry);
//     }

//     // Storage strategy based on size
//     if (value_len <= MAX_DIRECT_CACHE_SIZE) {
//         // Direct memory storage
//         cache_entry->data.direct.ptr = palloc(value_len);
//         memcpy(cache_entry->data.direct.ptr, value, value_len);
//         cache_entry->data.direct.size = value_len;
//         cache_entry->storage_method = STORAGE_DIRECT_MEMORY;
//     }
//     else {
//         // Memory-mapped file storage
//         if (!generate_optimized_filename(filename, sizeof(filename))) {
//             pfree(key_arg);
//             pfree(value_arg);
//             PG_RETURN_BOOL(false);
//         }

//         int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0600);
//         if (fd == -1) {
//             CACHE_LOG(ERROR, "Failed to create cache file");
//             pfree(key_arg);
//             pfree(value_arg);
//             PG_RETURN_BOOL(false);
//         }

//         write(fd, value, value_len);
        
//         void *mmap_ptr = mmap(NULL, value_len, PROT_READ, MAP_SHARED, fd, 0);
        
//         if (mmap_ptr == MAP_FAILED) {
//             close(fd);
//             unlink(filename);
//             CACHE_LOG(ERROR, "Memory mapping failed");
//             pfree(key_arg);
//             pfree(value_arg);
//             PG_RETURN_BOOL(false);
//         }

//         cache_entry->data.mapped.fd = fd;
//         cache_entry->data.mapped.mmap_ptr = mmap_ptr;
//         cache_entry->data.mapped.size = value_len;
//         cache_entry->storage_method = 
//             (value_len <= MAX_MEDIUM_CACHE_SIZE) ? 
//             STORAGE_MMAP_FILE : STORAGE_LARGE_FILE;
//     }

//     // Set metadata
//     memcpy(cache_entry->key, key, key_len);
//     cache_entry->key[key_len] = '\0';
    
//     cache_entry->is_active = true;
//     cache_entry->hash_key = ultra_optimized_hash(key, key_len);
//     cache_entry->creation_time = get_precise_timestamp();

//     // Update total cache size
//     pg_atomic_fetch_add_u64(&UltraCache.total_cache_size, value_len);

//     // Cleanup
//     pfree(key_arg);
//     pfree(value_arg);

//     PG_RETURN_BOOL(true);
// }

// // Cache Get Function// Cache Get Function
// PG_FUNCTION_INFO_V1(ultra_micro_cache_get);
// Datum ultra_micro_cache_get(PG_FUNCTION_ARGS)
// {
//     text *key_arg;
//     char *key;
//     size_t key_len;
//     text *result;
//     OptimizedCacheEntry *cache_entry;
//     void *value_ptr = NULL;
//     size_t value_size = 0;

//     // Null input check
//     if (PG_ARGISNULL(0))
//         PG_RETURN_NULL();

//     // Ensure cache is initialized
//     if (!UltraCache.is_initialized) 
//         ultra_micro_cache_init(fcinfo);

//     // Process key
//     key_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
//     key = (char *)VARDATA_ANY(key_arg);
//     key_len = VARSIZE_ANY_EXHDR(key_arg);

//     // Input validation
//     if (key_len >= MAX_KEY_LENGTH) {
//         pfree(key_arg);
//         PG_RETURN_NULL();
//     }

//     // Quick lookup
//     uint32_t hash_index = ultra_optimized_hash(key, key_len);
//     cache_entry = &UltraCache.entries[hash_index];

//     // Validate entry
//     if (!cache_entry->is_active || 
//         strcmp(cache_entry->key, key) != 0) {
//         pfree(key_arg);
//         PG_RETURN_NULL();
//     }

//     // Check entry expiration
//     uint64_t current_time = get_precise_timestamp();
//     if (current_time - cache_entry->creation_time > 
//         CACHE_ENTRY_LIFETIME * 1000000000ULL) {
//         cleanup_cache_entry(cache_entry);
//         pfree(key_arg);
//         PG_RETURN_NULL();
//     }

//     // Retrieve based on storage method
//     switch (cache_entry->storage_method) {
//         case STORAGE_DIRECT_MEMORY:
//             value_ptr = cache_entry->data.direct.ptr;
//             value_size = cache_entry->data.direct.size;
//             break;

//         case STORAGE_MMAP_FILE:
//             value_ptr = cache_entry->data.mapped.mmap_ptr;
//             value_size = cache_entry->data.mapped.size;
//             break;

//         case STORAGE_LARGE_FILE:
//             // Pre-allocated buffer for quick reads
//             pg_atomic_test_set_flag(&QuickAccessBuffer.lock);
//             {
//                 int fd = cache_entry->data.mapped.fd;
//                 ssize_t bytes_read = pread(fd, QuickAccessBuffer.buffer, 
//                                            MAX_STREAM_BUFFER_SIZE, 0);
                
//                 if (bytes_read > 0) {
//                     value_ptr = QuickAccessBuffer.buffer;
//                     value_size = bytes_read;
//                 }
//             }
//             pg_atomic_unlocked_test_flag(&QuickAccessBuffer.lock);
//             break;
//     }

//     // Construct result using VARHDRSZ and SET_VARSIZE
//     result = (text *)palloc(VARHDRSZ + value_size);
//     SET_VARSIZE(result, VARHDRSZ + value_size);
//     memcpy(VARDATA(result), value_ptr, value_size);

//     pfree(key_arg);

//     PG_RETURN_BYTEA_P(result);
// }
// // Cache Delete Function
// PG_FUNCTION_INFO_V1(ultra_micro_cache_delete);
// Datum ultra_micro_cache_delete(PG_FUNCTION_ARGS)
// {
//     text *key_arg;
//     char *key;
//     size_t key_len;
//     uint32_t hash_index;
//     OptimizedCacheEntry *cache_entry;

//     // Validate input argument
//     if (PG_ARGISNULL(0))
//         PG_RETURN_BOOL(false);

//     // Ensure cache is initialized
//     if (!UltraCache.is_initialized) 
//         ultra_micro_cache_init(fcinfo);

//     // Process key with proper detoasting
//     key_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
//     key = (char *)VARDATA_ANY(key_arg);
//     key_len = VARSIZE_ANY_EXHDR(key_arg);

//     // Validate key length
//     if (key_len >= MAX_KEY_LENGTH) {
//         pfree(key_arg);
//         PG_RETURN_BOOL(false);
//     }

//     // Compute hash and get cache entry
//     hash_index = ultra_optimized_hash(key, key_len);
//     cache_entry = &UltraCache.entries[hash_index];

//     // Validate and delete entry
//     if (cache_entry->is_active && 
//         strcmp(cache_entry->key, key) == 0) {
        
//         // Adjust total cache size atomically
//         pg_atomic_fetch_sub_u64(&UltraCache.total_cache_size, 
//             (cache_entry->storage_method == STORAGE_DIRECT_MEMORY) ? 
//             cache_entry->data.direct.size : 
//             cache_entry->data.mapped.size);

//         // Cleanup entry
//         cleanup_cache_entry(cache_entry);
        
//         pfree(key_arg);
//         PG_RETURN_BOOL(true);
//     }

//     pfree(key_arg);
//     PG_RETURN_BOOL(false);
// }
// // Cache Clear Function

// PG_FUNCTION_INFO_V1(ultra_micro_cache_cleanup);
// Datum ultra_micro_cache_cleanup(PG_FUNCTION_ARGS)
// {
//     // Check if cache is initialized
//     if (!UltraCache.is_initialized) {
//         PG_RETURN_BOOL(false);
//     }

//     // Cleanup all cache entries
//     for (int i = 0; i < MAX_CACHE_ENTRIES; i++) {
//         cleanup_cache_entry(&UltraCache.entries[i]);
//     }

//     // Reset atomic counters and mark as cleaned up
//     pg_atomic_init_u64(&UltraCache.total_cache_size, 0);

//     // Optional: Free the entries memory if needed
//     if (UltraCache.entries) {
//         pfree(UltraCache.entries);
//         UltraCache.entries = NULL;
//     }

//     // Mark cache as not initialized for potential re-initialization
//     UltraCache.is_initialized = false;

//     PG_RETURN_BOOL(true);
// } 
// // Cache Stats Function


























// #define _GNU_SOURCE
// #define _POSIX_C_SOURCE 200809L
// #define __USE_XOPEN2K      

// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "utils/geo_decls.h"
// #include "utils/palloc.h"
// #include "utils/memutils.h"
// #include "access/htup_details.h"
// #include "storage/fd.h"
// #include "miscadmin.h"
// #include "port/atomics.h"

// #include <sys/mman.h>
// #include <fcntl.h>
// #include <unistd.h>
// #include <time.h>
// #include <string.h>
// #include <sys/uio.h>
// #include <sys/stat.h>
// #include <errno.h>
// #include <stdint.h>  // Add this for uint64_t, uint32_t definitions

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// // Enhanced Cache Configuration
// #define MAX_CACHE_ENTRIES 128
// #define MAX_KEY_LENGTH 256
// #define MAX_FILENAME_LEN 512
// #define MMAP_FILE_PREFIX "/dev/shm/pg_ultra_cache_"
// #define CACHE_ENTRY_LIFETIME 3600  // 1 hour in seconds
// #define MAX_DIRECT_CACHE_SIZE (4LL * 1024 * 1024)  // 4 MB direct cache
// #define MAX_MEDIUM_CACHE_SIZE (4LL * 1024 * 1024 * 1024)  // 4 GB medium cache
// #define MAX_STREAM_BUFFER_SIZE (64 * 1024)  // 64 KB streaming buffer
// #define MAX_LARGE_FILE_SIZE (2LL * 1024 * 1024 * 1024)  // 2 GB max file size


// // Improved logging macro
// #define CACHE_LOG(level, ...) 
//     do { 
//         elog(level, "UltraCache: " __VA_ARGS__); 
//     } while (0)

// // Storage Method Enum
// typedef enum {
//     STORAGE_DIRECT_MEMORY,
//     STORAGE_MMAP_FILE,
//     STORAGE_LARGE_FILE
// } storage_method_t;

// // Advanced Cache Entry Structure
// typedef struct __attribute__((packed)) {
//     char key[MAX_KEY_LENGTH];
//     char *value_path;
//     size_t value_size;
//     int mmap_fd;
//     void *mmap_ptr;
//     uint32_t hash_key;
//     uint64_t creation_time;
//     bool is_active;
//     storage_method_t storage_method;
// } UltraLargeCacheEntry;

// // Optimized Global Cache Structure
// static struct {
//     UltraLargeCacheEntry *entries;
//     MemoryContext context;
//     volatile int access_counter;
//     volatile bool is_initialized;
//     char base_cache_dir[MAX_FILENAME_LEN];
//     pg_atomic_uint64 total_cache_size;
// } UltraCache = {0};

// // Efficient Hash Function (FNV-1a variant)
// static inline uint32_t 
// ultra_optimized_hash(const char *key, size_t len) 
// {
//     uint32_t hash = 2166136261U;
//     for (size_t i = 0; i < len; i++) {
//         hash ^= (unsigned char)key[i];
//         hash *= 16777619U;
//     }
//     return hash % MAX_CACHE_ENTRIES;
// }

// // Precise Timestamp Function
// static inline uint64_t 
// get_precise_timestamp(void) 
// {
//     struct timespec ts;
//     if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
//         CACHE_LOG(WARNING, "Failed to get timestamp");
//         return 0;
//     }
//     return (ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
// }

// // Optimized Filename Generation with Error Handling
// static bool
// generate_optimized_filename(char *filename, size_t max_len) 
// {
//     int unique_id = __sync_fetch_and_add(&UltraCache.access_counter, 1);
//     uint64_t timestamp = get_precise_timestamp();
    
//     int result = snprintf(filename, max_len, "%s/%d_%d_%lu.cache", 
//                  UltraCache.base_cache_dir, 
//                  MyProcPid, 
//                  unique_id,
//                  timestamp);
    
//     if (result < 0 || (size_t)result >= max_len) {
//         CACHE_LOG(ERROR, "Failed to generate filename");
//         return false;
//     }
    
//     return true;
// }

// static bool initialize_cache_directory(void);

// // Enhanced Cache Entry Cleanup
// static void
// cleanup_cache_entry(UltraLargeCacheEntry *entry) 
// {
//     if (!entry->is_active) return;

//     // Detailed error logging for cleanup
//     switch (entry->storage_method) {
//         case STORAGE_DIRECT_MEMORY:
//             if (entry->mmap_ptr) {
//                 pfree(entry->mmap_ptr);
//             }
//             break;
//         case STORAGE_MMAP_FILE:
//             if (entry->mmap_ptr != MAP_FAILED && entry->mmap_ptr) {
//                 if (munmap(entry->mmap_ptr, entry->value_size) != 0) {
//                     CACHE_LOG(WARNING, "Failed to unmap memory: ");
//                 }
//             }
//             break;
//         case STORAGE_LARGE_FILE:
//             // No special unmapping needed
//             break;
//     }

//     // Close and remove file descriptor
//     if (entry->mmap_fd != -1) {
//         if (close(entry->mmap_fd) != 0) {
//             CACHE_LOG(WARNING, "Failed to close file: ");
//         }
//     }

//     // Remove temporary file with error handling
//     if (entry->value_path) {
//         if (unlink(entry->value_path) != 0 && errno != ENOENT) {
//             CACHE_LOG(WARNING, "Failed to remove cache file:");
//         }
//         pfree(entry->value_path);
//     }

//     // Zero-memory reset
//     memset(entry, 0, sizeof(UltraLargeCacheEntry));
// }

// // Initialize Base Cache Directory with Enhanced Error Handling
// PG_FUNCTION_INFO_V1(ultra_micro_cache_init);
// Datum ultra_micro_cache_init(PG_FUNCTION_ARGS)
// {
//     MemoryContext old_context;

//     // Atomic initialization with full barrier
//     if (__sync_bool_compare_and_swap(&UltraCache.is_initialized, false, true)) {
//         // Initialize base cache directory
//         if (!initialize_cache_directory()) {
//             UltraCache.is_initialized = false;
//             PG_RETURN_BOOL(false);
//         }

//         // Create memory context for cache management
//         UltraCache.context = AllocSetContextCreate(
//             TopMemoryContext,
//             "UltraLargeCache",
//             ALLOCSET_SMALL_SIZES
//         );

//         old_context = MemoryContextSwitchTo(UltraCache.context);
        
//         // Allocate entries with zero initialization
//         UltraCache.entries = palloc0(sizeof(UltraLargeCacheEntry) * MAX_CACHE_ENTRIES);
        
//         MemoryContextSwitchTo(old_context);

//         // Pre-initialize entry states
//         for (int i = 0; i < MAX_CACHE_ENTRIES; i++) {
//             UltraCache.entries[i].mmap_fd = -1;
//         }

//         // Initialize atomic cache size tracker
//         pg_atomic_init_u64(&UltraCache.total_cache_size, 0);
//     }

//     PG_RETURN_BOOL(true);
// }

// // Initialize Base Cache Directory with Enhanced Error Handling
// static bool
// initialize_cache_directory(void) 
// {
//     // Use process ID to create unique cache directory
//     int result = snprintf(UltraCache.base_cache_dir, 
//                           sizeof(UltraCache.base_cache_dir), 
//                           "/tmp/pg_ultra_cache_%d", 
//                           MyProcPid);
    
//     if (result < 0 || (size_t)result >= sizeof(UltraCache.base_cache_dir)) {
//         CACHE_LOG(ERROR, "Failed to generate cache directory path");
//         return false;
//     }
    
//     // Create directory with strict permissions
//     if (mkdir(UltraCache.base_cache_dir, 0700) != 0) {
//         if (errno != EEXIST) {
//             CACHE_LOG(WARNING, "Could not create ultra cache directory");
//             return false;
//         }
//     }
    
//     return true;
// }


// // Improved Cache Set Function with Better Error Handling
// PG_FUNCTION_INFO_V1(ultra_micro_cache_set);
// Datum ultra_micro_cache_set(PG_FUNCTION_ARGS)
// {
//     text *key_arg, *value_arg;
//     char *key, *value;
//     size_t key_len, value_len;  // Use size_t for lengths
//     uint32_t hash_index;
//     char filename[MAX_FILENAME_LEN];
//     int mmap_fd = -1;
//     void *mmap_ptr = MAP_FAILED;
//     UltraLargeCacheEntry *cache_entry;
//     ssize_t bytes_written;  // Declare before the block
    
//     // Fast null checks
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
//         PG_RETURN_BOOL(false);

//     // Ensure cache is initialized
//     if (!UltraCache.is_initialized) 
//         ultra_micro_cache_init(fcinfo);

//     // Efficient argument processing
//     key_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
//     value_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(1));

//     key = VARDATA_ANY(key_arg);
//     value = VARDATA_ANY(value_arg);
//     key_len = VARSIZE_ANY_EXHDR(key_arg);
//     value_len = VARSIZE_ANY_EXHDR(value_arg);

//     // Input validation
//     if (key_len >= MAX_KEY_LENGTH || value_len > MAX_LARGE_FILE_SIZE) {
//         pfree(key_arg);
//         pfree(value_arg);
//         CACHE_LOG(NOTICE, "Key or value size exceeds limits");
//         PG_RETURN_BOOL(false);
//     }

//     // Compute hash index
//     hash_index = ultra_optimized_hash(key, key_len);
//     cache_entry = &UltraCache.entries[hash_index];

//     // Quick entry cleanup if active
//     if (cache_entry->is_active) {
//         cleanup_cache_entry(cache_entry);
//     }

//     // Determine storage method based on value size
//     if (value_len <= MAX_DIRECT_CACHE_SIZE) {
//         // Direct memory storage for small values
//         cache_entry->mmap_ptr = palloc(value_len);
//         if (!cache_entry->mmap_ptr) {
//             CACHE_LOG(ERROR, "Memory allocation failed");
//             pfree(key_arg);
//             pfree(value_arg);
//             PG_RETURN_BOOL(false);
//         }
//         memcpy(cache_entry->mmap_ptr, value, value_len);
//         cache_entry->storage_method = STORAGE_DIRECT_MEMORY;
//     } 
//     else if (value_len <= MAX_MEDIUM_CACHE_SIZE) {
//         // Memory-mapped file for medium-sized values
//         if (!generate_optimized_filename(filename, sizeof(filename))) {
//             pfree(key_arg);
//             pfree(value_arg);
//             PG_RETURN_BOOL(false);
//         }
        
//         mmap_fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0600);
//         if (mmap_fd == -1) {
//             CACHE_LOG(ERROR, "Failed to create mmap file:");
//             pfree(key_arg);
//             pfree(value_arg);
//             PG_RETURN_BOOL(false);
//         }

//         if (ftruncate(mmap_fd, value_len) == -1) {
//             CACHE_LOG(ERROR, "File truncate failed: ");
//             close(mmap_fd);
//             unlink(filename);
//             pfree(key_arg);
//             pfree(value_arg);
//             PG_RETURN_BOOL(false);
//         }

//         mmap_ptr = mmap(NULL, value_len, PROT_WRITE, 
//                         MAP_SHARED | MAP_POPULATE, 
//                         mmap_fd, 0);
        
//         if (mmap_ptr == MAP_FAILED) {
//             CACHE_LOG(ERROR, "Memory mapping failed:");
//             close(mmap_fd);
//             unlink(filename);
//             pfree(key_arg);
//             pfree(value_arg);
//             PG_RETURN_BOOL(false);
//         }

//         memcpy(mmap_ptr, value, value_len);
//         msync(mmap_ptr, value_len, MS_ASYNC);

//         cache_entry->mmap_ptr = mmap_ptr;
//         cache_entry->mmap_fd = mmap_fd;
//         cache_entry->storage_method = STORAGE_MMAP_FILE;
//         cache_entry->value_path = pstrdup(filename);
//     } 
//     else {
//         // Large file storage for very large values
//         if (!generate_optimized_filename(filename, sizeof(filename))) {
//             pfree(key_arg);
//             pfree(value_arg);
//             PG_RETURN_BOOL(false);
//         }
        
//         mmap_fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0600);
//         if (mmap_fd == -1) {
//             CACHE_LOG(ERROR, "Failed to create large file: ");
//             pfree(key_arg);
//             pfree(value_arg);
//             PG_RETURN_BOOL(false);
//         }

//         // Write entire value to file with error checking
//             bytes_written = write(mmap_fd, value, value_len);
//             if (bytes_written != (ssize_t)value_len) {
//                 CACHE_LOG(ERROR, "Incomplete file write:");
//                 close(mmap_fd);
//                 unlink(filename);
//                 pfree(key_arg);
//                 pfree(value_arg);
//                 PG_RETURN_BOOL(false);
//             }

//         cache_entry->mmap_ptr = NULL;
//         cache_entry->mmap_fd = mmap_fd;
//         cache_entry->storage_method = STORAGE_LARGE_FILE;
//         cache_entry->value_path = pstrdup(filename);
//     }

//     // Store entry metadata
//     memset(cache_entry->key, 0, MAX_KEY_LENGTH);
//     memcpy(cache_entry->key, key, key_len);
    
//     cache_entry->is_active = true;
//     cache_entry->value_size = value_len;
//     cache_entry->hash_key = ultra_optimized_hash(key, key_len);
//     cache_entry->creation_time = get_precise_timestamp();

//     // Update total cache size atomically
//     pg_atomic_fetch_add_u64(&UltraCache.total_cache_size, value_len);

//     // Clean up temporary allocations
//     pfree(key_arg);
//     pfree(value_arg);

//     PG_RETURN_BOOL(true);
// }


// // Retrieve a value by key from the cache

// PG_FUNCTION_INFO_V1(ultra_micro_cache_get);
// Datum ultra_micro_cache_get(PG_FUNCTION_ARGS)
// {
//     text *key_arg;
//     char *key;
//     size_t key_len;
//     uint32_t hash_index;
//     UltraLargeCacheEntry *cache_entry;
//     text *result;
//     // Declare all variables at the top of the function
//     uint64_t current_time;
//     void *value_ptr;
//     size_t value_size;
//     ssize_t bytes_read;
//     int fd;

//     // Ensure cache is initialized
//     if (!UltraCache.is_initialized) 
//         ultra_micro_cache_init(fcinfo);

//     // Check for null input
//     if (PG_ARGISNULL(0))
//         PG_RETURN_NULL();

//     // Process input key
//     key_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
//     key = VARDATA_ANY(key_arg);
//     key_len = VARSIZE_ANY_EXHDR(key_arg);

//     // Input validation
//     if (key_len >= MAX_KEY_LENGTH) {
//         pfree(key_arg);
//         CACHE_LOG(NOTICE, "Key size exceeds limit");
//         PG_RETURN_NULL();
//     }

//     // Compute hash index
//     hash_index = ultra_optimized_hash(key, key_len);
//     cache_entry = &UltraCache.entries[hash_index];

//     // Check entry validity
//     if (!cache_entry->is_active || 
//         strncmp(cache_entry->key, key, key_len) != 0) {
//         pfree(key_arg);
//         PG_RETURN_NULL();
//     }

//     // Check cache entry expiration (1 hour lifetime)
//     current_time = get_precise_timestamp();
//     if (current_time - cache_entry->creation_time > CACHE_ENTRY_LIFETIME * 1000000000ULL) {
//         cleanup_cache_entry(cache_entry);
//         pfree(key_arg);
//         PG_RETURN_NULL();
//     }

//     // Retrieve value based on storage method
//     value_ptr = NULL;
//     value_size = 0;

//     switch (cache_entry->storage_method) {
//         case STORAGE_DIRECT_MEMORY:
//             value_ptr = cache_entry->mmap_ptr;
//             value_size = cache_entry->value_size;
//             break;
//         case STORAGE_MMAP_FILE:
//             value_ptr = cache_entry->mmap_ptr;
//             value_size = cache_entry->value_size;
//             break;
//         case STORAGE_LARGE_FILE:
//             // For large files, read from file
//             if (cache_entry->value_path) {
//                 fd = open(cache_entry->value_path, O_RDONLY);
//                 if (fd == -1) {
//                     CACHE_LOG(WARNING, "Failed to open large file cache");
//                     pfree(key_arg);
//                     PG_RETURN_NULL();
//                 }
                
//                 value_ptr = palloc(cache_entry->value_size);
//                 bytes_read = read(fd, value_ptr, cache_entry->value_size);
//                 close(fd);

//                 if (bytes_read != (ssize_t)cache_entry->value_size) {
//                     CACHE_LOG(WARNING, "Failed to read large file cache");
//                     pfree(value_ptr);
//                     pfree(key_arg);
//                     PG_RETURN_NULL();
//                 }
//                 value_size = cache_entry->value_size;
//                 break;
//             }
//             break;
//     }

//     // Construct result text
//     result = (text *)palloc(VARHDRSZ + value_size);
//     SET_VARSIZE(result, VARHDRSZ + value_size);
//     memcpy(VARDATA(result), value_ptr, value_size);

//     // Cleanup for direct and mmap methods
//     if (cache_entry->storage_method == STORAGE_LARGE_FILE) {
//         pfree(value_ptr);
//     }

//     pfree(key_arg);
//     PG_RETURN_DATUM(PointerGetDatum(result));
// }

// // Delete a specific key from the cache
// PG_FUNCTION_INFO_V1(ultra_micro_cache_delete);
// Datum ultra_micro_cache_delete(PG_FUNCTION_ARGS)
// {
//     text *key_arg;
//     char *key;
//     size_t key_len;
//     uint32_t hash_index;
//     UltraLargeCacheEntry *cache_entry;

//     // Ensure cache is initialized
//     if (!UltraCache.is_initialized) 
//         ultra_micro_cache_init(fcinfo);

//     // Check for null input
//     if (PG_ARGISNULL(0))
//         PG_RETURN_BOOL(false);

//     // Process input key
//     key_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
//     key = VARDATA_ANY(key_arg);
//     key_len = VARSIZE_ANY_EXHDR(key_arg);

//     // Input validation
//     if (key_len >= MAX_KEY_LENGTH) {
//         pfree(key_arg);
//         CACHE_LOG(NOTICE, "Key size exceeds limit");
//         PG_RETURN_BOOL(false);
//     }

//     // Compute hash index
//     hash_index = ultra_optimized_hash(key, key_len);
//     cache_entry = &UltraCache.entries[hash_index];

//     // Check entry validity
//     if (!cache_entry->is_active || 
//         strncmp(cache_entry->key, key, key_len) != 0) {
//         pfree(key_arg);
//         PG_RETURN_BOOL(false);
//     }

//     // Cleanup the cache entry
//     cleanup_cache_entry(cache_entry);

//     // Decrement total cache size
//     pg_atomic_fetch_sub_u64(&UltraCache.total_cache_size, cache_entry->value_size);

//     pfree(key_arg);
//     PG_RETURN_BOOL(true);
// }

// // Cleanup expired entries from the cache
// PG_FUNCTION_INFO_V1(ultra_micro_cache_cleanup);
// Datum ultra_micro_cache_cleanup(PG_FUNCTION_ARGS)
// {
//     uint64_t current_time = get_precise_timestamp();
//     int cleaned_entries = 0;

//     // Ensure cache is initialized
//     if (!UltraCache.is_initialized) 
//         ultra_micro_cache_init(fcinfo);

//     // Iterate through all cache entries
//     for (int i = 0; i < MAX_CACHE_ENTRIES; i++) {
//         UltraLargeCacheEntry *cache_entry = &UltraCache.entries[i];

//         // Check if entry is active and expired
//         if (cache_entry->is_active && 
//             (current_time - cache_entry->creation_time > CACHE_ENTRY_LIFETIME * 1000000000ULL)) {
            
//             // Log the cleanup
//             CACHE_LOG(NOTICE, "Cleaning up expired cache entry: %s", cache_entry->key);

//             // Decrement total cache size
//             pg_atomic_fetch_sub_u64(&UltraCache.total_cache_size, cache_entry->value_size);

//             // Cleanup the entry
//             cleanup_cache_entry(cache_entry);
            
//             cleaned_entries++;
//         }
//     }

//     // Log total cleaned entries
//     CACHE_LOG(NOTICE, "Cleaned %d expired cache entries", cleaned_entries);

//     PG_RETURN_BOOL(true);
// }













































//do this

// #define _GNU_SOURCE         // Most comprehensive feature test macro
// #define _POSIX_C_SOURCE 200809L  // For POSIX.1-2008 features
// #define __USE_XOPEN2K      
// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "utils/geo_decls.h"
// #include "utils/palloc.h"
// #include "utils/memutils.h"
// #include "access/htup_details.h"
// #include "storage/fd.h"
// #include "miscadmin.h"
// #include <sys/mman.h>
// #include <fcntl.h>
// #include <unistd.h>
// #include <time.h>
// #include <string.h>
// #include <sys/uio.h> 

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif


// // Hyper-Optimized Cache Configuration
// #define MAX_CACHE_ENTRIES 32        // Increased for better hash distribution
// #define MAX_KEY_LENGTH 48            // Reduced to optimize memory
// #define MAX_FILENAME_LEN 192         // More unique filename space
// #define MMAP_FILE_PREFIX "/dev/shm/pg_ultra_cache_"  // Use RAM disk for faster I/O
// #define CACHE_ENTRY_LIFETIME 60    // Reduced to 30 minutes

// // Advanced Cache Entry Structure with Memory Alignment
// typedef struct __attribute__((packed)) {
//     char key[MAX_KEY_LENGTH];        // Compact key storage
//     char *value_path;                // Pointer to memory-mapped file path
//     size_t value_size;               // Actual value size
//     int mmap_fd;                     // File descriptor for memory mapping
//     void *mmap_ptr;                  // Pointer to memory-mapped data
//     uint32_t hash_key;               // Pre-computed hash for faster lookup
//     uint64_t creation_time;          // Nanosecond timestamp for precision
//     bool is_active;                  // Compact flag for entry status
// } UltraMicroCacheEntry;

// // Optimized Global Cache Structure
// static struct {
//     UltraMicroCacheEntry *entries;   // Dynamic entry array
//     MemoryContext context;           // Dedicated memory context
//     volatile int access_counter;     // Thread-safe access counter
//     volatile bool is_initialized;    // Atomic initialization flag
// } UltraCache = {0};

// // High-Performance Hash Function (xxHash-inspired)
// static inline uint32_t 
// ultra_optimized_hash(const char *key, size_t len) 
// {
//     uint32_t hash = 2166136261U;  // FNV-1a initial basis
//     for (size_t i = 0; i < len; i++) {
//         hash ^= key[i];
//         hash *= 16777619U;  // FNV prime
//     }
//     return hash % MAX_CACHE_ENTRIES;
// }

// // Nanosecond Precise Timestamp
// static inline uint64_t 
// get_precise_timestamp(void) 
// {
//     struct timespec ts;
//     // Use CLOCK_MONOTONIC if CLOCK_MONOTONIC_RAW is not available
//     clock_gettime(CLOCK_MONOTONIC, &ts);
//     return (ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
// }

// // Ultra-Fast Filename Generation
// static void
// generate_optimized_filename(char *filename, size_t max_len) 
// {
//     // Use atomic increment for thread safety
//     int unique_id = __sync_fetch_and_add(&UltraCache.access_counter, 1);
//     uint64_t timestamp = get_precise_timestamp();
    
//     snprintf(filename, max_len, "%s%d_%d_%lu", 
//              MMAP_FILE_PREFIX, 
//              MyProcPid, 
//              unique_id,
//              timestamp);
// }
// // Efficient Cache Entry Cleanup
// static void
// cleanup_cache_entry(UltraMicroCacheEntry *entry) 
// {
//     if (!entry->is_active) return;

//     // Quick atomic memory unmapping
//     if (entry->mmap_ptr && entry->value_size) {
//         munmap(entry->mmap_ptr, entry->value_size);
//     }

//     // Rapid file descriptor management
//     if (entry->mmap_fd != -1) {
//         close(entry->mmap_fd);
//     }

//     // Quick file removal
//     if (entry->value_path) {
//         unlink(entry->value_path);
//     }

//     // Zero-memory reset
//     memset(entry, 0, sizeof(UltraMicroCacheEntry));
// }

// // Highly Optimized Cache Initialization
// PG_FUNCTION_INFO_V1(ultra_micro_cache_init);
// Datum ultra_micro_cache_init(PG_FUNCTION_ARGS)
// {
//     MemoryContext old_context;

//     // Atomic check to prevent multiple initializations
//     if (__sync_bool_compare_and_swap(&UltraCache.is_initialized, false, true)) {
//         // Create memory context with minimal overhead
//         UltraCache.context = AllocSetContextCreate(
//             TopMemoryContext,
//             "UltraFastCache",
//             ALLOCSET_SMALL_SIZES
//         );

//         // Allocate entries in one go
//         old_context = MemoryContextSwitchTo(UltraCache.context);
//         UltraCache.entries = palloc0(sizeof(UltraMicroCacheEntry) * MAX_CACHE_ENTRIES);
//         MemoryContextSwitchTo(old_context);

//         // Pre-initialize entry states
//         for (int i = 0; i < MAX_CACHE_ENTRIES; i++) {
//             UltraCache.entries[i].mmap_fd = -1;
//         }
//     }

//     PG_RETURN_BOOL(true);
// }



// // Optimized Cache Set Function
// PG_FUNCTION_INFO_V1(ultra_micro_cache_set);
// Datum ultra_micro_cache_set(PG_FUNCTION_ARGS)
// {
//     text *key_arg, *value_arg;
//     char *key;
//     char *value;
//     int32 key_len;
//     int32 value_len;
//     uint32_t hash_index;
//     char filename[MAX_FILENAME_LEN];
//     int mmap_fd = -1;
//     void *mmap_ptr = MAP_FAILED;
//     uint32_t pre_computed_hash;
//     UltraMicroCacheEntry *cache_entry;
    
//     // Fast null checks
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
//         PG_RETURN_BOOL(false);

//     // Ensure cache is initialized
//     if (!UltraCache.is_initialized) 
//         ultra_micro_cache_init(fcinfo);

//     // Efficient argument processing
//     key_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
//     value_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(1));

//     key = VARDATA_ANY(key_arg);
//     value = VARDATA_ANY(value_arg);
//     key_len = VARSIZE_ANY_EXHDR(key_arg);
//     value_len = VARSIZE_ANY_EXHDR(value_arg);

//     // Rapid input validation
//     if (key_len >= MAX_KEY_LENGTH) {
//         pfree(key_arg);
//         pfree(value_arg);
//         PG_RETURN_BOOL(false);
//     }

//     // Compute hash index
//     hash_index = ultra_optimized_hash(key, key_len);
//     pre_computed_hash = ultra_optimized_hash(key, key_len);

//     // Quick entry cleanup
//     if (UltraCache.entries[hash_index].is_active) {
//         cleanup_cache_entry(&UltraCache.entries[hash_index]);
//     }

//     // Generate unique filename using RAM disk
//     generate_optimized_filename(filename, sizeof(filename));
//     mmap_fd = open(filename, O_RDWR | O_CREAT | O_TRUNC | PG_O_DIRECT, 0600);
    
//     if (mmap_fd == -1) {
//         pfree(key_arg);
//         pfree(value_arg);
//         PG_RETURN_BOOL(false);
//     }

//     // Efficient memory mapping
//     if (ftruncate(mmap_fd, value_len) == -1 ||
//     (mmap_ptr = mmap(NULL, value_len, PROT_WRITE, 
//                      MAP_SHARED | MAP_POPULATE, 
//                      mmap_fd, 0)) == MAP_FAILED) {
//         close(mmap_fd);
//         unlink(filename);
//         pfree(key_arg);
//         pfree(value_arg);
//         PG_RETURN_BOOL(false);
//     }

//     // Ultra-fast memory copy with prefetching
//     __builtin_prefetch(mmap_ptr, 1, 1);
//     memcpy(mmap_ptr, value, value_len);
//     msync(mmap_ptr, value_len, MS_ASYNC);

//     // Compact entry storage
//     cache_entry = &UltraCache.entries[hash_index];
//     memset(cache_entry->key, 0, MAX_KEY_LENGTH);
//     memcpy(cache_entry->key, key, key_len);
    
//     cache_entry->value_path = pstrdup(filename);
//     cache_entry->value_size = value_len;
//     cache_entry->mmap_fd = mmap_fd;
//     cache_entry->mmap_ptr = mmap_ptr;
//     cache_entry->is_active = true;
//     cache_entry->creation_time = get_precise_timestamp();
//     cache_entry->hash_key = pre_computed_hash;

//     pfree(key_arg);
//     pfree(value_arg);

//     PG_RETURN_BOOL(true);
// }

// PG_FUNCTION_INFO_V1(ultra_micro_cache_get);
// Datum ultra_micro_cache_get(PG_FUNCTION_ARGS)
// {
//     text *key_arg;
//     char *key;
//     int32 key_len;
//     uint32_t hash_index;
//     text *result_text;
//     UltraMicroCacheEntry *cache_entry;

//     if (PG_ARGISNULL(0) || !UltraCache.is_initialized)
//         PG_RETURN_NULL();

//     key_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
//     key = VARDATA_ANY(key_arg);
//     key_len = VARSIZE_ANY_EXHDR(key_arg);

//     // Simplified hash computation
//     hash_index = ultra_optimized_hash(key, key_len);
//     cache_entry = &UltraCache.entries[hash_index];

//     // Ultra-fast, minimal comparison
//     if (cache_entry->is_active &&
//         strlen(cache_entry->key) == key_len &&
//         memcmp(cache_entry->key, key, key_len) == 0) {
        
//         result_text = cstring_to_text_with_len(
//             cache_entry->mmap_ptr,
//             cache_entry->value_size
//         );

//         pfree(key_arg);
//         PG_RETURN_TEXT_P(result_text);
//     }

//     pfree(key_arg);
//     PG_RETURN_NULL();
// }

// // Efficient Cleanup Function
// PG_FUNCTION_INFO_V1(ultra_micro_cache_cleanup);
// Datum ultra_micro_cache_cleanup(PG_FUNCTION_ARGS)
// {
//     uint64_t current_time = get_precise_timestamp();

//     if (!UltraCache.is_initialized)
//         PG_RETURN_BOOL(false);

//     for (int i = 0; i < MAX_CACHE_ENTRIES; i++) {
//         if (UltraCache.entries[i].is_active &&
//             (current_time - UltraCache.entries[i].creation_time) > 
//             (CACHE_ENTRY_LIFETIME * 1000000000ULL)) {
//             cleanup_cache_entry(&UltraCache.entries[i]);
//         }
//     }

//     PG_RETURN_BOOL(true);
// }

// // Optimized Shutdown Function
// PG_FUNCTION_INFO_V1(ultra_micro_cache_shutdown);
// Datum ultra_micro_cache_shutdown(PG_FUNCTION_ARGS)
// {
//     if (UltraCache.is_initialized) {
//         for (int i = 0; i < MAX_CACHE_ENTRIES; i++) {
//             cleanup_cache_entry(&UltraCache.entries[i]);
//         }

//         pfree(UltraCache.entries);
//         MemoryContextDelete(UltraCache.context);
        
//         // Reset global structure
//         memset(&UltraCache, 0, sizeof(UltraCache));
//     }

//     PG_RETURN_BOOL(true);
// }








































// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "utils/geo_decls.h"
// #include "utils/palloc.h"
// #include "utils/memutils.h"
// #include "access/htup_details.h"
// #include "storage/fd.h"
// #include "miscadmin.h"
// #include <sys/mman.h>
// #include <fcntl.h>
// #include <unistd.h>
// #include <time.h>

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif


// #define TUPTOASTER_H
// // Optimized cache configuration
// #define MAX_CACHE_ENTRIES 16
// #define MAX_KEY_LENGTH 64
// #define MAX_FILENAME_LEN 256
// #define MMAP_FILE_PREFIX "/tmp/pg_ultra_cache_"
// #define CACHE_ENTRY_LIFETIME 3600  // 1 hour cache lifetime

// // Advanced Cache Entry Structure
// typedef struct {
//     char key[MAX_KEY_LENGTH];      // Fixed-size key
//     char *value_path;              // Path to memory-mapped file
//     size_t value_size;             // Actual value size
//     int mmap_fd;                   // File descriptor for memory mapping
//     void *mmap_ptr;                // Pointer to memory-mapped data
//     bool is_active;                // Flag to track active entries
//     time_t creation_time;          // Timestamp of entry creation
//     MemoryContext entry_context;   // Memory context for this entry
// } UltraMicroCacheEntry;

// // Global cache structure with memory context
// static UltraMicroCacheEntry *ultra_micro_cache = NULL;
// static MemoryContext cache_memory_context = NULL;
// static int current_entry_index = 0;
// static int cache_access_counter = 0;

// // Simple, fast hash function (unchanged from previous implementation)
// static inline uint32_t
// ultra_simple_hash(const char *key, size_t len)
// {
//     uint32_t hash = 2166136261U;
//     size_t i;
//     for (i = 0; i < len; i++) {
//         hash = (hash * 16777619U) ^ key[i];
//     }
//     return hash % MAX_CACHE_ENTRIES;
// }

// // Generate unique temporary filename
// static void
// generate_unique_filename(char *filename, size_t max_len)
// {
//     snprintf(filename, max_len, "%s%d_%d", 
//              MMAP_FILE_PREFIX, 
//              MyProcPid, 
//              cache_access_counter++);
// }

// // Improved cleanup for a single cache entry
// static void
// cleanup_cache_entry(UltraMicroCacheEntry *entry)
// {
//     // Safely unmap memory
//     if (entry->mmap_ptr != NULL && entry->value_size > 0) {
//         munmap(entry->mmap_ptr, entry->value_size);
//         entry->mmap_ptr = NULL;
//     }

//     // Close and reset file descriptor
//     if (entry->mmap_fd != -1) {
//         close(entry->mmap_fd);
//         entry->mmap_fd = -1;
//     }

//     // Remove temporary file
//     if (entry->value_path != NULL) {
//         unlink(entry->value_path);
//     }

//     // Reset entry state
//     memset(entry->key, 0, MAX_KEY_LENGTH);
//     entry->value_size = 0;
//     entry->is_active = false;
//     entry->creation_time = 0;

//     // Destroy the entry's memory context
//     if (entry->entry_context != NULL) {
//         MemoryContextDelete(entry->entry_context);
//         entry->entry_context = NULL;
//     }
// }

// // Ultra-lightweight cache initialization
// PG_FUNCTION_INFO_V1(ultra_micro_cache_init);
// Datum ultra_micro_cache_init(PG_FUNCTION_ARGS)
// {
//     // Create a dedicated memory context for the cache
//     if (cache_memory_context == NULL) {
//         cache_memory_context = AllocSetContextCreate(
//             TopMemoryContext,
//             "UltraMicroCache",
//             ALLOCSET_SMALL_SIZES
//         );
//     }

//     // Allocate cache entries in the memory context
//     if (ultra_micro_cache == NULL) {
//         MemoryContext old_context = MemoryContextSwitchTo(cache_memory_context);
        
//         ultra_micro_cache = palloc0(sizeof(UltraMicroCacheEntry) * MAX_CACHE_ENTRIES);
        
//         // Initialize each entry
//         for (int i = 0; i < MAX_CACHE_ENTRIES; i++) {
//             ultra_micro_cache[i].mmap_fd = -1;
//             ultra_micro_cache[i].entry_context = AllocSetContextCreate(
//                 cache_memory_context,
//                 "CacheEntryContext",
//                 ALLOCSET_SMALL_SIZES
//             );
//         }
        
//         MemoryContextSwitchTo(old_context);
//     }

//     PG_RETURN_BOOL(true);
// }

// // Optimized cache set function
// PG_FUNCTION_INFO_V1(ultra_micro_cache_set);
// Datum ultra_micro_cache_set(PG_FUNCTION_ARGS)
// {
//     // Similar implementation to previous version, but with memory context management
//     text *key_arg;
//     text *value_arg;
//     char *key;
//     char *value;
//     int32 key_len;
//     int32 value_len;
//     uint32_t hash_index;
//     char filename[MAX_FILENAME_LEN];
//     int mmap_fd = -1;
//     void *mmap_ptr = NULL;

//     // Null checks
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
//         PG_RETURN_BOOL(false);

//     // Ensure cache is initialized
//     if (ultra_micro_cache == NULL) {
//         ultra_micro_cache_init(fcinfo);
//     }

//     // Fetch arguments and process similar to previous implementation
//     key_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
//     value_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(1));

//     key = VARDATA_ANY(key_arg);
//     value = VARDATA_ANY(value_arg);
//     key_len = VARSIZE_ANY_EXHDR(key_arg);
//     value_len = VARSIZE_ANY_EXHDR(value_arg);

//     // Validate input sizes
//     if (key_len >= MAX_KEY_LENGTH) {
//         pfree(key_arg);
//         pfree(value_arg);
//         PG_RETURN_BOOL(false);
//     }

//     // Compute hash index
//     hash_index = ultra_simple_hash(key, key_len);

//     // Cleanup any existing entry
//     if (ultra_micro_cache[hash_index].is_active) {
//         cleanup_cache_entry(&ultra_micro_cache[hash_index]);
//     }

//     // Switch to entry's memory context for allocation
//     MemoryContext old_context = MemoryContextSwitchTo(
//         ultra_micro_cache[hash_index].entry_context
//     );

//     // Generate unique filename and create memory-mapped file
//     generate_unique_filename(filename, sizeof(filename));
//     mmap_fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0600);
    
//     if (mmap_fd == -1) {
//         MemoryContextSwitchTo(old_context);
//         pfree(key_arg);
//         pfree(value_arg);
//         PG_RETURN_BOOL(false);
//     }

//     // Extend file and memory map
//     if (ftruncate(mmap_fd, value_len) == -1 ||
//         (mmap_ptr = mmap(NULL, value_len, PROT_WRITE, MAP_SHARED, mmap_fd, 0)) == MAP_FAILED) {
//         close(mmap_fd);
//         unlink(filename);
//         MemoryContextSwitchTo(old_context);
//         pfree(key_arg);
//         pfree(value_arg);
//         PG_RETURN_BOOL(false);
//     }

//     // Copy value to memory-mapped file
//     memcpy(mmap_ptr, value, value_len);
//     msync(mmap_ptr, value_len, MS_SYNC);

//     // Store cache entry details
//     memset(ultra_micro_cache[hash_index].key, 0, MAX_KEY_LENGTH);
//     memcpy(ultra_micro_cache[hash_index].key, key, key_len);
    
//     ultra_micro_cache[hash_index].value_path = pstrdup(filename);
//     ultra_micro_cache[hash_index].value_size = value_len;
//     ultra_micro_cache[hash_index].mmap_fd = mmap_fd;
//     ultra_micro_cache[hash_index].mmap_ptr = mmap_ptr;
//     ultra_micro_cache[hash_index].is_active = true;
//     ultra_micro_cache[hash_index].creation_time = time(NULL);

//     // Restore previous memory context
//     MemoryContextSwitchTo(old_context);

//     // Free detoasted datums
//     pfree(key_arg);
//     pfree(value_arg);

//     PG_RETURN_BOOL(true);
// }

// // Get function (similar to previous implementation)
// PG_FUNCTION_INFO_V1(ultra_micro_cache_get);
// Datum ultra_micro_cache_get(PG_FUNCTION_ARGS)
// {
//     // Most of the implementation remains the same
//     text *key_arg;
//     char *key;
//     int32 key_len;
//     uint32_t hash_index;
//     text *result_text;
//     time_t current_time;

//     // Null and initialization checks
//     if (PG_ARGISNULL(0) || ultra_micro_cache == NULL)
//         PG_RETURN_NULL();

//     key_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
//     key = VARDATA_ANY(key_arg);
//     key_len = VARSIZE_ANY_EXHDR(key_arg);

//     // Compute hash index
//     hash_index = ultra_simple_hash(key, key_len);

//     // Check entry validity and age
//     current_time = time(NULL);
//     if (ultra_micro_cache[hash_index].is_active &&
//         (current_time - ultra_micro_cache[hash_index].creation_time) <= CACHE_ENTRY_LIFETIME &&
//         strlen(ultra_micro_cache[hash_index].key) == key_len &&
//         memcmp(ultra_micro_cache[hash_index].key, key, key_len) == 0) {
        
//         result_text = cstring_to_text_with_len(
//             ultra_micro_cache[hash_index].mmap_ptr,
//             ultra_micro_cache[hash_index].value_size
//         );

//         pfree(key_arg);
//         PG_RETURN_TEXT_P(result_text);
//     }

//     pfree(key_arg);
//     PG_RETURN_NULL();
// }

// // Improved cleanup function
// PG_FUNCTION_INFO_V1(ultra_micro_cache_cleanup);
// Datum ultra_micro_cache_cleanup(PG_FUNCTION_ARGS)
// {
//     time_t current_time;

//     // Ensure cache is initialized and not NULL
//     if (ultra_micro_cache == NULL || cache_memory_context == NULL)
//         PG_RETURN_BOOL(false);

//     current_time = time(NULL);

//     // Iterate through entries and cleanup expired ones
//     for (int i = 0; i < MAX_CACHE_ENTRIES; i++) {
//         if (ultra_micro_cache[i].is_active &&
//             (current_time - ultra_micro_cache[i].creation_time) > CACHE_ENTRY_LIFETIME) {
//             cleanup_cache_entry(&ultra_micro_cache[i]);
//         }
//     }

//     PG_RETURN_BOOL(true);
// }

// // Optional: Shutdown function to completely free resources
// PG_FUNCTION_INFO_V1(ultra_micro_cache_shutdown);
// Datum ultra_micro_cache_shutdown(PG_FUNCTION_ARGS)
// {
//     if (ultra_micro_cache != NULL) {
//         // Cleanup all entries
//         for (int i = 0; i < MAX_CACHE_ENTRIES; i++) {
//             cleanup_cache_entry(&ultra_micro_cache[i]);
//         }

//         // Free the cache array
//         pfree(ultra_micro_cache);
//         ultra_micro_cache = NULL;
//     }

//     // Delete the memory context
//     if (cache_memory_context != NULL) {
//         MemoryContextDelete(cache_memory_context);
//         cache_memory_context = NULL;
//     }

//     PG_RETURN_BOOL(true);
// }

























































































// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "utils/geo_decls.h"
// #include "utils/palloc.h"
// #include "access/htup_details.h"
// #include "storage/bufmgr.h"
// #include "storage/fd.h"
// #include "miscadmin.h"
// #include <sys/mman.h>
// #include <fcntl.h>
// #include <unistd.h>

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// #define TUPTOASTER_H
// // Highly optimized cache configuration
// #define MAX_CACHE_ENTRIES 16
// #define MAX_KEY_LENGTH 64
// #define MAX_FILENAME_LEN 256
// #define MMAP_FILE_PREFIX "/tmp/pg_ultra_cache_"

// // Advanced Cache Entry Structure
// typedef struct {
//     char key[MAX_KEY_LENGTH];      // Fixed-size key to avoid dynamic allocation
//     char *value_path;              // Path to memory-mapped file
//     size_t value_size;             // Actual value size
//     int mmap_fd;                   // File descriptor for memory mapping
//     void *mmap_ptr;                // Pointer to memory-mapped data
//     bool is_active;                // Flag to track active entries
// } UltraMicroCacheEntry;

// // Global cache structure
// static UltraMicroCacheEntry ultra_micro_cache[MAX_CACHE_ENTRIES];
// static int current_entry_index = 0;
// static int cache_access_counter = 0;

// // Simple, fast hash function
// static inline uint32_t
// ultra_simple_hash(const char *key, size_t len)
// {
//     uint32_t hash = 2166136261U;
//     size_t i;
//     for (i = 0; i < len; i++) {
//         hash = (hash * 16777619U) ^ key[i];
//     }
//     return hash % MAX_CACHE_ENTRIES;
// }

// // Create unique temporary filename
// static void
// generate_unique_filename(char *filename, size_t max_len)
// {
//     snprintf(filename, max_len, "%s%d_%d", 
//              MMAP_FILE_PREFIX, 
//              MyProcPid, 
//              cache_access_counter++);
// }

// // Cleanup a single cache entry
// static void
// cleanup_cache_entry(UltraMicroCacheEntry *entry)
// {
//     // Safely cleanup memory-mapped resources
//     if (entry->mmap_ptr != NULL && entry->value_size > 0) {
//         munmap(entry->mmap_ptr, entry->value_size);
//         entry->mmap_ptr = NULL;
//     }

//     // Close file descriptor if valid
//     if (entry->mmap_fd != -1) {
//         close(entry->mmap_fd);
//         entry->mmap_fd = -1;
//     }

//     // Remove temporary file if path exists
//     if (entry->value_path != NULL) {
//         unlink(entry->value_path);
//         pfree(entry->value_path);
//         entry->value_path = NULL;
//     }

//     // Reset other fields
//     memset(entry->key, 0, MAX_KEY_LENGTH);
//     entry->value_size = 0;
//     entry->is_active = false;
// }

// // Ultra-lightweight cache initialization
// PG_FUNCTION_INFO_V1(ultra_micro_cache_init);
// Datum ultra_micro_cache_init(PG_FUNCTION_ARGS)
// {
//     size_t i;
//     for (i = 0; i < MAX_CACHE_ENTRIES; i++) {
//         // Initialize all entries to a safe state
//         memset(ultra_micro_cache[i].key, 0, MAX_KEY_LENGTH);
//         ultra_micro_cache[i].value_path = NULL;
//         ultra_micro_cache[i].mmap_ptr = NULL;
//         ultra_micro_cache[i].mmap_fd = -1;
//         ultra_micro_cache[i].value_size = 0;
//         ultra_micro_cache[i].is_active = false;
//     }
//     current_entry_index = 0;
//     PG_RETURN_BOOL(true);
// }

// // Optimized cache set function
// PG_FUNCTION_INFO_V1(ultra_micro_cache_set);
// Datum ultra_micro_cache_set(PG_FUNCTION_ARGS)
// {
//     text *key_arg;
//     text *value_arg;
//     char *key;
//     char *value;
//     int32 key_len;
//     int32 value_len;
//     uint32_t hash_index;
//     char filename[MAX_FILENAME_LEN];
//     int mmap_fd = -1;
//     void *mmap_ptr = NULL;

//     // Null checks with early return
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
//         PG_RETURN_BOOL(false);

//     // Fetch arguments
//     key_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
//     value_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(1));

//     // Get key and value details
//     key = VARDATA_ANY(key_arg);
//     value = VARDATA_ANY(value_arg);
//     key_len = VARSIZE_ANY_EXHDR(key_arg);
//     value_len = VARSIZE_ANY_EXHDR(value_arg);

//     // Validate input sizes
//     if (key_len >= MAX_KEY_LENGTH) {
//         pfree(key_arg);
//         pfree(value_arg);
//         PG_RETURN_BOOL(false);
//     }

//     // Compute hash index
//     hash_index = ultra_simple_hash(key, key_len);

//     // Cleanup any existing entry at this index
//     if (ultra_micro_cache[hash_index].is_active) {
//         cleanup_cache_entry(&ultra_micro_cache[hash_index]);
//     }

//     // Generate unique filename for memory mapping
//     generate_unique_filename(filename, sizeof(filename));

//     // Create memory-mapped file
//     mmap_fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0600);
//     if (mmap_fd == -1) {
//         pfree(key_arg);
//         pfree(value_arg);
//         PG_RETURN_BOOL(false);
//     }

//     // Extend file to required size
//     if (ftruncate(mmap_fd, value_len) == -1) {
//         close(mmap_fd);
//         unlink(filename);
//         pfree(key_arg);
//         pfree(value_arg);
//         PG_RETURN_BOOL(false);
//     }

//     // Memory map the file
//     mmap_ptr = mmap(NULL, value_len, PROT_WRITE, MAP_SHARED, mmap_fd, 0);
//     if (mmap_ptr == MAP_FAILED) {
//         close(mmap_fd);
//         unlink(filename);
//         pfree(key_arg);
//         pfree(value_arg);
//         PG_RETURN_BOOL(false);
//     }

//     // Copy value to memory-mapped file
//     memcpy(mmap_ptr, value, value_len);
//     msync(mmap_ptr, value_len, MS_SYNC);

//     // Store cache entry
//     memset(ultra_micro_cache[hash_index].key, 0, MAX_KEY_LENGTH);
//     memcpy(ultra_micro_cache[hash_index].key, key, key_len);
    
//     // Allocate and store filename
//     ultra_micro_cache[hash_index].value_path = palloc(MAX_FILENAME_LEN);
//     strcpy(ultra_micro_cache[hash_index].value_path, filename);
    
//     // Store other metadata
//     ultra_micro_cache[hash_index].value_size = value_len;
//     ultra_micro_cache[hash_index].mmap_fd = mmap_fd;
//     ultra_micro_cache[hash_index].mmap_ptr = mmap_ptr;
//     ultra_micro_cache[hash_index].is_active = true;

//     // Free detoasted datums
//     pfree(key_arg);
//     pfree(value_arg);

//     PG_RETURN_BOOL(true);
// }

// // Ultra-fast get function
// PG_FUNCTION_INFO_V1(ultra_micro_cache_get);
// Datum ultra_micro_cache_get(PG_FUNCTION_ARGS)
// {
//     text *key_arg;
//     char *key;
//     int32 key_len;
//     uint32_t hash_index;
//     text *result_text;

//     // Fast null check
//     if (PG_ARGISNULL(0))
//         PG_RETURN_NULL();

//     // Fetch datum to ensure correct typing
//     key_arg = PG_DETOAST_DATUM(PG_GETARG_DATUM(0));

//     // Get key details
//     key = VARDATA_ANY(key_arg);
//     key_len = VARSIZE_ANY_EXHDR(key_arg);

//     // Compute hash index
//     hash_index = ultra_simple_hash(key, key_len);

//     // Check if entry exists and matches
//     if (ultra_micro_cache[hash_index].is_active &&
//         strlen(ultra_micro_cache[hash_index].key) == key_len &&
//         memcmp(ultra_micro_cache[hash_index].key, key, key_len) == 0) {
//         // Create text directly
//         result_text = cstring_to_text_with_len(
//             ultra_micro_cache[hash_index].mmap_ptr,
//             ultra_micro_cache[hash_index].value_size
//         );

//         // Free detoasted datum
//         pfree(key_arg);

//         PG_RETURN_TEXT_P(result_text);
//     }

//     // Free detoasted datum
//     pfree(key_arg);

//     PG_RETURN_NULL();
// }

// // Minimal cleanup function
// PG_FUNCTION_INFO_V1(ultra_micro_cache_cleanup);
// Datum ultra_micro_cache_cleanup(PG_FUNCTION_ARGS)
// {
//     size_t i;
//     for (i = 0; i < MAX_CACHE_ENTRIES; i++) {
//         // Only cleanup active entries
//         if (ultra_micro_cache[i].is_active) {
//             cleanup_cache_entry(&ultra_micro_cache[i]);
//         }
//     }
//     current_entry_index = 0;
//     PG_RETURN_BOOL(true);
// }










































// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "utils/geo_decls.h"  // Include this for VARDATA and VARSIZE macros
// #include "utils/palloc.h"
// #include "access/htup_details.h"

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// #define TUPTOASTER_H
// // Ultra-compact cache configuration
// #define MAX_CACHE_ENTRIES 16
// #define MAX_KEY_LENGTH 64
// #define MAX_VALUE_LENGTH 1024

// // Minimal Cache Entry Structure
// typedef struct {
//     char key[MAX_KEY_LENGTH];
//     char value[MAX_VALUE_LENGTH];
//     bool is_active;
// } UltraMicroCacheEntry;

// // Static global cache to avoid dynamic allocation
// static UltraMicroCacheEntry ultra_micro_cache[MAX_CACHE_ENTRIES];
// static int current_entry_index = 0;

// // Simple, fast hash function
// static inline uint32_t
// ultra_simple_hash(const char *key, size_t len)
// {
//     uint32_t hash = 2166136261U;
//     for (size_t i = 0; i < len; i++) {
//         hash = (hash * 16777619U) ^ key[i];
//     }
//     return hash % MAX_CACHE_ENTRIES;
// }

// // Ultra-lightweight cache initialization
// PG_FUNCTION_INFO_V1(ultra_micro_cache_init);
// Datum ultra_micro_cache_init(PG_FUNCTION_ARGS)
// {
//     // Simple memset to zero out the cache
//     memset(ultra_micro_cache, 0, sizeof(ultra_micro_cache));
//     current_entry_index = 0;
//     PG_RETURN_BOOL(true);
// }

// // Optimized cache set function
// PG_FUNCTION_INFO_V1(ultra_micro_cache_set);
// Datum ultra_micro_cache_set(PG_FUNCTION_ARGS)
// {
//     text *key_arg;
//     text *value_arg;
//     char *key;
//     char *value;
//     int32 key_len;
//     int32 value_len;
//     uint32_t hash_index;
//     Datum key_datum;
//     Datum value_datum;

//     // Null checks with early return
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
//         PG_RETURN_BOOL(false);

//     // Fetch datums to ensure correct typing
//     key_datum = PG_GETARG_DATUM(0);
//     value_datum = PG_GETARG_DATUM(1);

//     // Fetch arguments
//     key_arg = PG_DETOAST_DATUM_SLICE(key_datum, 0, -1);
//     value_arg = PG_DETOAST_DATUM_SLICE(value_datum, 0, -1);

//     // Get key and value details
//     key = VARDATA_ANY(key_arg);
//     value = VARDATA_ANY(value_arg);
//     key_len = VARSIZE_ANY_EXHDR(key_arg);
//     value_len = VARSIZE_ANY_EXHDR(value_arg);

//     // Validate input sizes
//     if (key_len >= MAX_KEY_LENGTH || value_len >= MAX_VALUE_LENGTH) {
//         // Free detoasted datums
//         if (key_arg != (text *)key_datum)
//             pfree(key_arg);
//         if (value_arg != (text *)value_datum)
//             pfree(value_arg);
//         PG_RETURN_BOOL(false);
//     }

//     // Compute hash index
//     hash_index = ultra_simple_hash(key, key_len);

//     // Copy key and value
//     memcpy(ultra_micro_cache[hash_index].key, key, key_len);
//     ultra_micro_cache[hash_index].key[key_len] = '\0';
//     memcpy(ultra_micro_cache[hash_index].value, value, value_len);
//     ultra_micro_cache[hash_index].value[value_len] = '\0';
//     ultra_micro_cache[hash_index].is_active = true;

//     // Free detoasted datums if necessary
//     if (key_arg != (text *)key_datum)
//         pfree(key_arg);
//     if (value_arg != (text *)value_datum)
//         pfree(value_arg);

//     PG_RETURN_BOOL(true);
// }

// // Ultra-fast get function
// PG_FUNCTION_INFO_V1(ultra_micro_cache_get);
// Datum ultra_micro_cache_get(PG_FUNCTION_ARGS)
// {
//     text *key_arg;
//     char *key;
//     int32 key_len;
//     uint32_t hash_index;
//     text *result_text;
//     Datum key_datum;

//     // Fast null check
//     if (PG_ARGISNULL(0))
//         PG_RETURN_NULL();

//     // Fetch datum to ensure correct typing
//     key_datum = PG_GETARG_DATUM(0);

//     // Ensure toasted argument is detoasted
//     key_arg = PG_DETOAST_DATUM_SLICE(key_datum, 0, -1);

//     // Get key details
//     key = VARDATA_ANY(key_arg);
//     key_len = VARSIZE_ANY_EXHDR(key_arg);

//     // Compute hash index
//     hash_index = ultra_simple_hash(key, key_len);

//     // Check if entry exists and matches
//     if (ultra_micro_cache[hash_index].is_active &&
//         strlen(ultra_micro_cache[hash_index].key) == key_len &&
//         memcmp(ultra_micro_cache[hash_index].key, key, key_len) == 0) {
//         // Create text directly
//         result_text = cstring_to_text_with_len(
//             ultra_micro_cache[hash_index].value,
//             strlen(ultra_micro_cache[hash_index].value)
//         );

//         // Free detoasted datum if necessary
//         if (key_arg != (text *)key_datum)
//             pfree(key_arg);

//         PG_RETURN_TEXT_P(result_text);
//     }

//     // Free detoasted datum if necessary
//     if (key_arg != (text *)key_datum)
//         pfree(key_arg);

//     PG_RETURN_NULL();
// }

// // Minimal cleanup function
// PG_FUNCTION_INFO_V1(ultra_micro_cache_cleanup);
// Datum ultra_micro_cache_cleanup(PG_FUNCTION_ARGS)
// {
//     // Simply reset the cache
//     memset(ultra_micro_cache, 0, sizeof(ultra_micro_cache));
//     current_entry_index = 0;
//     PG_RETURN_BOOL(true);
// }


//large 


// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "storage/shmem.h"
// #include "utils/memutils.h"
// #include "access/hash.h"
// #include <string.h>
// #include <stdint.h>
// #include <time.h>

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// #define CACHE_ENTRIES 4096       // Power of 2 for faster modulo
// #define ENTRY_LIFETIME 3600      // 1 hour default expiry
// #define MAX_KEY_LENGTH 64        // Reduced key length for faster comparison
// #define MAX_SEGMENTS 256         // Maximum number of segments for a single key
// #define SEGMENT_LENGTH 4096      // Increased segment size for larger texts

// // Hyper-optimized cache entry for large text segmentation
// typedef struct {
//     uint64_t hash;               // Precomputed hash
//     uint64_t expiry;             // Expiry timestamp
//     char key[MAX_KEY_LENGTH];    // Compact key storage
//     char segment[SEGMENT_LENGTH]; // Segment of large text
//     uint16_t segment_index;      // Segment index
//     uint16_t total_segments;     // Total segments for this key
//     bool valid;                  // Valid flag
// } __attribute__((packed)) LargeTextCacheEntry;

// // Cache structure with minimal overhead
// typedef struct {
//     LargeTextCacheEntry entries[CACHE_ENTRIES];
//     uint64_t last_access;        // Timestamp of last cache operation
// } LargeTextCache;

// // Global cache instance - placed in a performance-critical memory region
// static LargeTextCache *large_text_cache = NULL;

// // Extremely fast hash function (xxHash-inspired)
// static inline __attribute__((always_inline)) uint64_t 
// large_text_hash(const char *key) {
//     uint64_t h = 0xCBF29CE484222325ULL;
    
//     while (*key) {
//         h ^= *key++;
//         h *= 0x100000001B3ULL;
//         h ^= h >> 33;
//     }
    
//     return h;
// }

// // Initialize large text cache
// PG_FUNCTION_INFO_V1(large_text_cache_init);
// Datum large_text_cache_init(PG_FUNCTION_ARGS)
// {
//     // All variables declared at the top for C90 compatibility
//     int i;

//     if (large_text_cache == NULL) {
//         // Allocate cache in TopMemoryContext for persistence
//         large_text_cache = MemoryContextAllocZero(TopMemoryContext, sizeof(LargeTextCache));
        
//         // Preemptively zero all entries and set valid to false
//         for (i = 0; i < CACHE_ENTRIES; i++) {
//             large_text_cache->entries[i].valid = false;
//         }
        
//         // Set initial last access timestamp
//         large_text_cache->last_access = time(NULL);
//     }
    
//     PG_RETURN_BOOL(true);
// }

// // Large text set operation
// PG_FUNCTION_INFO_V1(large_text_cache_set);
// Datum large_text_cache_set(PG_FUNCTION_ARGS)
// {
//     // Declare all variables at the top
//     text *key_arg;
//     text *value_arg;
//     char *key;
//     char *value;
//     size_t value_length;
//     uint64_t hash;
//     uint16_t total_segments;
//     bool set_success;
//     int i, seg;
//     LargeTextCacheEntry *entry;
//     size_t copy_length;
//     bool found_slot;

//     if (large_text_cache == NULL || PG_ARGISNULL(0) || PG_ARGISNULL(1)) {
//         elog(ERROR, "Cache not initialized or invalid input");
//         PG_RETURN_BOOL(false);
//     }

//     // Get input arguments
//     key_arg = PG_GETARG_TEXT_PP(0);
//     value_arg = PG_GETARG_TEXT_PP(1);
    
//     // Convert to C strings
//     key = text_to_cstring(key_arg);
//     value = text_to_cstring(value_arg);
//     value_length = strlen(value);
    
//     // Compute hash
//     hash = large_text_hash(key);
    
//     // Calculate total segments needed
//     total_segments = (value_length + SEGMENT_LENGTH - 1) / SEGMENT_LENGTH;
//     if (total_segments > MAX_SEGMENTS) {
//         elog(ERROR, "Text exceeds maximum supported size");
//         pfree(key);
//         pfree(value);
//         PG_RETURN_BOOL(false);
//     }
    
//     // Remove any existing segments for this key first
//     for (i = 0; i < CACHE_ENTRIES; i++) {
//         entry = &large_text_cache->entries[i];
//         if (entry->valid && 
//             entry->hash == hash && 
//             strcmp(entry->key, key) == 0) {
//             entry->valid = false;
//         }
//     }
    
//     // Store segments
//     set_success = true;
//     for (seg = 0; seg < total_segments; seg++) {
//         // Find an available slot
//         found_slot = false;
//         for (i = 0; i < CACHE_ENTRIES; i++) {
//             entry = &large_text_cache->entries[i];
            
//             if (!entry->valid) {
//                 // Copy key
//                 strncpy(entry->key, key, MAX_KEY_LENGTH - 1);
//                 entry->key[MAX_KEY_LENGTH - 1] = '\0';
                
//                 // Copy segment
//                 copy_length = (seg == total_segments - 1) ? 
//                     value_length % SEGMENT_LENGTH : SEGMENT_LENGTH;
//                 if (copy_length == 0) copy_length = SEGMENT_LENGTH;
                
//                 memcpy(entry->segment, 
//                        value + seg * SEGMENT_LENGTH, 
//                        copy_length);
//                 entry->segment[copy_length] = '\0';
                
//                 // Set metadata
//                 entry->hash = hash;
//                 entry->expiry = time(NULL) + ENTRY_LIFETIME;
//                 entry->segment_index = seg;
//                 entry->total_segments = total_segments;
//                 entry->valid = true;
                
//                 found_slot = true;
//                 break;
//             }
//         }
        
//         // If no slot found, mark as failure
//         if (!found_slot) {
//             set_success = false;
//             break;
//         }
//     }
    
//     // Free temporary strings
//     pfree(key);
//     pfree(value);
    
//     PG_RETURN_BOOL(set_success);
// }

// // Large text get operation
// PG_FUNCTION_INFO_V1(large_text_cache_get);
// Datum large_text_cache_get(PG_FUNCTION_ARGS)
// {
//     // Declare all variables at the top
//     text *key_arg;
//     char *key;
//     uint64_t hash;
//     uint64_t now;
//     uint16_t total_segments = 0;
//     uint16_t max_segment_index = 0;
//     int i, seg;
//     LargeTextCacheEntry *entry;
//     char *full_text;
//     text *result;
//     bool segment_found;

//     if (large_text_cache == NULL || PG_ARGISNULL(0)) {
//         elog(ERROR, "Cache not initialized or invalid input");
//         PG_RETURN_NULL();
//     }

//     // Get input key
//     key_arg = PG_GETARG_TEXT_PP(0);
//     key = text_to_cstring(key_arg);
    
//     // Compute hash
//     hash = large_text_hash(key);
    
//     // Search through entire cache
//     now = time(NULL);
    
//     // First, find the total number of segments
//     for (i = 0; i < CACHE_ENTRIES; i++) {
//         entry = &large_text_cache->entries[i];
        
//         if (entry->valid && 
//             entry->hash == hash && 
//             entry->expiry > now && 
//             strcmp(entry->key, key) == 0) 
//         {
//             total_segments = entry->total_segments;
//             max_segment_index = (entry->total_segments > max_segment_index) ? 
//                 entry->total_segments : max_segment_index;
//         }
//     }
    
//     // If no segments found, return NULL
//     if (total_segments == 0) {
//         pfree(key);
//         PG_RETURN_NULL();
//     }
    
//     // Reconstruct full text
//     full_text = palloc(total_segments * SEGMENT_LENGTH + 1);
//     full_text[0] = '\0';
    
//     // Collect all segments
//     for (seg = 0; seg < total_segments; seg++) {
//         segment_found = false;
//         for (i = 0; i < CACHE_ENTRIES; i++) {
//             entry = &large_text_cache->entries[i];
            
//             if (entry->valid && 
//                 entry->hash == hash && 
//                 entry->expiry > now && 
//                 strcmp(entry->key, key) == 0 && 
//                 entry->segment_index == seg) 
//             {
//                 strcat(full_text, entry->segment);
//                 segment_found = true;
//                 break;
//             }
//         }
        
//         // If any segment is missing, return NULL
//         if (!segment_found) {
//             pfree(full_text);
//             pfree(key);
//             PG_RETURN_NULL();
//         }
//     }
    
//     // Convert to PostgreSQL text type
//     result = cstring_to_text_with_len(full_text, strlen(full_text));
    
//     // Free temporary strings
//     pfree(full_text);
//     pfree(key);
    
//     PG_RETURN_TEXT_P(result);
// }

// // Vacuum operation (minimal overhead)
// PG_FUNCTION_INFO_V1(large_text_cache_vacuum);
// Datum large_text_cache_vacuum(PG_FUNCTION_ARGS)
// {
//     // Declare variables at the top
//     int removed_count;
//     uint64_t now;
//     int i;
//     LargeTextCacheEntry *entry;

//     if (large_text_cache == NULL) {
//         PG_RETURN_INT32(0);
//     }

//     removed_count = 0;
//     now = time(NULL);

//     // Rapid vacuum without context switching
//     for (i = 0; i < CACHE_ENTRIES; i++) {
//         entry = &large_text_cache->entries[i];
//         if (entry->valid &&
//             entry->expiry <= now) {
//             // Zero out expired entry
//             memset(entry, 0, sizeof(LargeTextCacheEntry));
//             removed_count++;
//         }
//     }
    
//     PG_RETURN_INT32(removed_count);
// }

// // Cleanup operation
// PG_FUNCTION_INFO_V1(large_text_cache_cleanup);
// Datum large_text_cache_cleanup(PG_FUNCTION_ARGS)
// {
//     if (large_text_cache != NULL) {
//         // Zero out entire cache
//         memset(large_text_cache, 0, sizeof(LargeTextCache));
        
//         // Free memory
//         pfree(large_text_cache);
//         large_text_cache = NULL;
//     }
    
//     PG_RETURN_BOOL(true);
// }














// // /the current best one

// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "storage/shmem.h"
// #include "utils/memutils.h"
// #include "access/hash.h"
// #include <string.h>
// #include <stdint.h>
// #include <time.h>

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// #define CACHE_ENTRIES 4096  // Power of 2 for faster modulo
// #define ENTRY_LIFETIME 3600 // 1 hour default expiry
// #define CACHE_LINE_SIZE 64  // Typical cache line size
// #define MAX_KEY_LENGTH 255
// #define MAX_VALUE_LENGTH 1023

// // Optimized cache entry with state management
// typedef struct {
//     uint64_t hash;           // Full 64-bit hash
//     bool is_valid;           // Entry validity flag
//     uint64_t expiry_time;    // Expiry timestamp
//     char key[MAX_KEY_LENGTH + 1];     // Null-terminated key storage
//     char value[MAX_VALUE_LENGTH + 1]; // Null-terminated value storage
// } OptimizedCacheEntry;

// typedef struct {
//     OptimizedCacheEntry entries[CACHE_ENTRIES];
//     MemoryContext context;
// } UltraFastCache;

// static UltraFastCache *ultra_cache = NULL;

// // Optimized hash function (MurmurHash3-inspired)
// static inline uint64_t fast_hash(const char *key) {
//     const uint64_t m = 0xc6a4a7935bd1e995ULL;
//     const int r = 47;
//     uint64_t h = 0x8445d61642c5cd10ULL ^ (strlen(key) * m);

//     while (*key) {
//         uint64_t k = *key++;
//         k *= m;
//         k ^= k >> r;
//         k *= m;
        
//         h ^= k;
//         h *= m;
//     }

//     h ^= h >> r;
//     h *= m;
//     h ^= h >> r;

//     return h;
// }

// // Initialize ultra-fast cache
// PG_FUNCTION_INFO_V1(ultra_cache_init);
// Datum ultra_cache_init(PG_FUNCTION_ARGS)
// {
//     if (ultra_cache == NULL) {
//         ultra_cache = MemoryContextAllocZero(TopMemoryContext, sizeof(UltraFastCache));
//         ultra_cache->context = AllocSetContextCreate(TopMemoryContext, 
//                                                      "UltraFastCacheContext",
//                                                      ALLOCSET_SMALL_SIZES);
        
//         // Zero-initialize all entries
//         memset(ultra_cache->entries, 0, sizeof(ultra_cache->entries));
//     }
//     PG_RETURN_BOOL(true);
// }

// // Ultra-fast set operation
// PG_FUNCTION_INFO_V1(ultra_cache_set);
// Datum ultra_cache_set(PG_FUNCTION_ARGS)
// {
//     if (ultra_cache == NULL || PG_ARGISNULL(0) || PG_ARGISNULL(1)) {
//         elog(ERROR, "Cache not initialized or invalid input");
//         PG_RETURN_BOOL(false);
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(ultra_cache->context);

//     PG_TRY();
//     {
//         text *key_arg = PG_GETARG_TEXT_PP(0);
//         text *value_arg = PG_GETARG_TEXT_PP(1);
        
//         char *key = text_to_cstring(key_arg);
//         char *value = text_to_cstring(value_arg);
        
//         uint64_t hash = fast_hash(key);
//         uint32_t index = hash & (CACHE_ENTRIES - 1);
        
//         OptimizedCacheEntry *entry = &ultra_cache->entries[index];
        
//         // Prepare entry
//         uint64_t expiry = time(NULL) + ENTRY_LIFETIME;
        
//         // Update entry
//         entry->hash = hash;
//         strncpy(entry->key, key, MAX_KEY_LENGTH);
//         entry->key[MAX_KEY_LENGTH] = '\0';
        
//         strncpy(entry->value, value, MAX_VALUE_LENGTH);
//         entry->value[MAX_VALUE_LENGTH] = '\0';
        
//         entry->expiry_time = expiry;
//         entry->is_valid = true;
        
//         pfree(key);
//         pfree(value);
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
//     PG_RETURN_BOOL(true);
// }

// // Ultra-fast get operation
// PG_FUNCTION_INFO_V1(ultra_cache_get);
// Datum ultra_cache_get(PG_FUNCTION_ARGS)
// {
//     if (ultra_cache == NULL || PG_ARGISNULL(0)) {
//         elog(ERROR, "Cache not initialized or invalid input");
//         PG_RETURN_NULL();
//     }

//     text *result = NULL;
//     time_t now = time(NULL);

//     MemoryContext oldcontext = MemoryContextSwitchTo(ultra_cache->context);

//     PG_TRY();
//     {
//         text *key_arg = PG_GETARG_TEXT_PP(0);
//         char *key = text_to_cstring(key_arg);
        
//         uint64_t hash = fast_hash(key);
//         uint32_t index = hash & (CACHE_ENTRIES - 1);
        
//         OptimizedCacheEntry *entry = &ultra_cache->entries[index];
        
//         // Quick check with minimal overhead
//         if (entry->is_valid && 
//             entry->hash == hash && 
//             entry->expiry_time > now &&
//             strcmp(entry->key, key) == 0) 
//         {
//             result = cstring_to_text(entry->value);
//         }
        
//         pfree(key);
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
    
//     if (result) {
//         PG_RETURN_TEXT_P(result);
//     }
//     PG_RETURN_NULL();
// }

// // Vacuum expired entries
// PG_FUNCTION_INFO_V1(ultra_cache_vacuum);
// Datum ultra_cache_vacuum(PG_FUNCTION_ARGS)
// {
//     if (ultra_cache == NULL) {
//         elog(ERROR, "Cache not initialized");
//         PG_RETURN_INT32(0);
//     }

//     int removed_count = 0;
//     time_t now = time(NULL);

//     MemoryContext oldcontext = MemoryContextSwitchTo(ultra_cache->context);

//     PG_TRY();
//     {
//         for (int i = 0; i < CACHE_ENTRIES; i++) {
//             OptimizedCacheEntry *entry = &ultra_cache->entries[i];
            
//             // Check if entry is expired
//             if (entry->is_valid && entry->expiry_time <= now) {
//                 // Reset entry
//                 entry->is_valid = false;
//                 entry->key[0] = '\0';
//                 entry->value[0] = '\0';
//                 removed_count++;
//             }
//         }
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
    
//     PG_RETURN_INT32(removed_count);
// }

// // Clean up cache
// PG_FUNCTION_INFO_V1(ultra_cache_cleanup);
// Datum ultra_cache_cleanup(PG_FUNCTION_ARGS)
// {
//     if (ultra_cache == NULL) {
//         PG_RETURN_BOOL(true);
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(TopMemoryContext);

//     PG_TRY();
//     {
//         // Simply invalidate all entries
//         for (int i = 0; i < CACHE_ENTRIES; i++) {
//             ultra_cache->entries[i].is_valid = false;
//         }
        
//         MemoryContextDelete(ultra_cache->context);
//         pfree(ultra_cache);
//         ultra_cache = NULL;
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
//     PG_RETURN_BOOL(true);
// }

















// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "storage/shmem.h"
// #include "utils/memutils.h"
// #include <string.h>
// #include <time.h>

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// // Optimized structure for caching
// typedef struct {
//     char *key;
//     char *value;
//     time_t expiry;
// } CacheEntry;

// // Global cache storage
// static CacheEntry *cache_entries = NULL;
// static int cache_capacity = 1000;  // Configurable capacity
// static int current_entries = 0;
// static MemoryContext cache_context = NULL;

// // Initialize the cache
// PG_FUNCTION_INFO_V1(cache_init);
// Datum cache_init(PG_FUNCTION_ARGS)
// {
//     // Create a dedicated memory context
//     if (cache_context == NULL) {
//         cache_context = AllocSetContextCreate(TopMemoryContext,
//                                               "OptimizedCacheContext",
//                                               ALLOCSET_SMALL_SIZES);
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_context);

//     PG_TRY();
//     {
//         // Allocate memory for cache entries if not exists
//         if (cache_entries == NULL) {
//             cache_entries = palloc0(sizeof(CacheEntry) * cache_capacity);
//             elog(INFO, "Cache initialized with %d capacity", cache_capacity);
//         }
        
//         // Reset current entries
//         current_entries = 0;
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
    
//     PG_RETURN_BOOL(true);
// }

// // Set a key-value pair in the cache
// PG_FUNCTION_INFO_V1(cache_set);
// Datum cache_set(PG_FUNCTION_ARGS)
// {
//     // Validate cache and inputs
//     if (cache_entries == NULL || PG_ARGISNULL(0) || PG_ARGISNULL(1)) {
//         elog(ERROR, "Cache not initialized or invalid input");
//         PG_RETURN_BOOL(false);
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_context);

//     PG_TRY();
//     {
//         // Get input parameters as text
//         text *key_arg = PG_GETARG_TEXT_PP(0);
//         text *value_arg = PG_GETARG_TEXT_PP(1);
        
//         // Convert text to C strings safely
//         char *key = text_to_cstring(key_arg);
//         char *value = text_to_cstring(value_arg);
        
//         // Check if cache is full
//         if (current_entries >= cache_capacity) {
//             elog(WARNING, "Cache is full, cannot add more entries");
//             pfree(key);
//             pfree(value);
//             PG_RETURN_BOOL(false);
//         }
        
//         // Find an existing entry or use a new slot
//         bool entry_found = false;
//         for (int i = 0; i < current_entries; i++) {
//             if (strcmp(cache_entries[i].key, key) == 0) {
//                 // Update existing entry
//                 pfree(cache_entries[i].value);
//                 cache_entries[i].value = pstrdup(value);
//                 cache_entries[i].expiry = time(NULL) + 3600;  // 1-hour expiry
//                 entry_found = true;
//                 break;
//             }
//         }
        
//         // If no existing entry found, add a new one
//         if (!entry_found) {
//             cache_entries[current_entries].key = pstrdup(key);
//             cache_entries[current_entries].value = pstrdup(value);
//             cache_entries[current_entries].expiry = time(NULL) + 3600;
//             current_entries++;
//         }
        
//         // Free temporary strings
//         pfree(key);
//         pfree(value);
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
    
//     PG_RETURN_BOOL(true);
// }

// // Get a value from the cache
// PG_FUNCTION_INFO_V1(cache_get);
// Datum cache_get(PG_FUNCTION_ARGS)
// {
//     // Validate cache and input
//     if (cache_entries == NULL || PG_ARGISNULL(0)) {
//         elog(ERROR, "Cache not initialized or invalid input");
//         PG_RETURN_NULL();
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_context);
//     text *result = NULL;

//     PG_TRY();
//     {
//         // Get input key
//         text *key_arg = PG_GETARG_TEXT_PP(0);
//         char *key = text_to_cstring(key_arg);
        
//         time_t now = time(NULL);
        
//         // Search for the key
//         for (int i = 0; i < current_entries; i++) {
//             if (strcmp(cache_entries[i].key, key) == 0) {
//                 // Check if entry has not expired
//                 if (now <= cache_entries[i].expiry) {
//                     result = cstring_to_text(cache_entries[i].value);
//                     break;
//                 }
//             }
//         }
        
//         // Free the key string
//         pfree(key);
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
    
//     // Return result (NULL if not found)
//     if (result) {
//         PG_RETURN_TEXT_P(result);
//     }
//     PG_RETURN_NULL();
// }

// // Vacuum function to remove expired entries
// PG_FUNCTION_INFO_V1(cache_vacuum);
// Datum cache_vacuum(PG_FUNCTION_ARGS)
// {
//     // Ensure cache is initialized
//     if (cache_entries == NULL) {
//         elog(ERROR, "Cache not initialized");
//         PG_RETURN_INT32(0);
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_context);
//     int removed_count = 0;

//     PG_TRY();
//     {
//         time_t now = time(NULL);
        
//         for (int i = 0; i < current_entries; i++) {
//             if (now > cache_entries[i].expiry) {
//                 // Free expired entry
//                 pfree(cache_entries[i].key);
//                 pfree(cache_entries[i].value);
                
//                 // Shift remaining entries
//                 for (int j = i; j < current_entries - 1; j++) {
//                     cache_entries[j] = cache_entries[j + 1];
//                 }
                
//                 current_entries--;
//                 removed_count++;
//                 i--;  // Recheck the next entry after shifting
//             }
//         }
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
    
//     PG_RETURN_INT32(removed_count);
// }

// // Cleanup function
// PG_FUNCTION_INFO_V1(cache_cleanup);
// Datum cache_cleanup(PG_FUNCTION_ARGS)
// {
//     // Prevent cleanup if no entries exist
//     if (cache_entries == NULL) {
//         PG_RETURN_BOOL(true);
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_context);

//     PG_TRY();
//     {
//         // Free each entry
//         for (int i = 0; i < current_entries; i++) {
//             if (cache_entries[i].key) {
//                 pfree(cache_entries[i].key);
//                 pfree(cache_entries[i].value);
//             }
//         }
        
//         // Free the cache entries array
//         pfree(cache_entries);
//         cache_entries = NULL;
        
//         // Reset entry count
//         current_entries = 0;
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
    
//     PG_RETURN_BOOL(true);
// }






// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "storage/shmem.h"
// #include "utils/memutils.h"
// #include <string.h>
// #include <time.h>

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// // Safer cache entry structure
// typedef struct {
//     char *key;
//     char *value;
//     time_t expiry;
// } CacheEntry;

// // Global cache storage with safer management
// static CacheEntry *cache_entries = NULL;
// static int cache_capacity = 100;  // Maximum number of entries
// static int current_entries = 0;

// // Memory context for safer allocation
// static MemoryContext cache_context = NULL;

// // Initialize the cache with error handling
// PG_FUNCTION_INFO_V1(cache_init);
// Datum cache_init(PG_FUNCTION_ARGS)
// {
//     // Create a dedicated memory context
//     if (cache_context == NULL) {
//         cache_context = AllocSetContextCreate(TopMemoryContext,
//                                               "CacheExtensionContext",
//                                               ALLOCSET_SMALL_SIZES);
//     }

//     // Switch to our memory context
//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_context);

//     PG_TRY();
//     {
//         // Allocate memory for cache entries
//         if (cache_entries == NULL) {
//             cache_entries = palloc0(sizeof(CacheEntry) * cache_capacity);
//             elog(INFO, "Cache initialized with %d capacity", cache_capacity);
//         }
//     }
//     PG_CATCH();
//     {
//         // Restore the original memory context
//         MemoryContextSwitchTo(oldcontext);
        
//         // Re-throw the error
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     // Restore the original memory context
//     MemoryContextSwitchTo(oldcontext);
    
//     PG_RETURN_BOOL(true);
// }

// // Set a key-value pair in the cache with extensive error checking
// PG_FUNCTION_INFO_V1(cache_set);
// Datum cache_set(PG_FUNCTION_ARGS)
// {
//     // Ensure cache is initialized
//     if (cache_entries == NULL) {
//         elog(ERROR, "Cache not initialized. Call cache_init() first.");
//         PG_RETURN_BOOL(false);
//     }

//     // Validate input parameters
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) {
//         elog(ERROR, "Key or value cannot be NULL");
//         PG_RETURN_BOOL(false);
//     }

//     // Switch to cache memory context
//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_context);

//     PG_TRY();
//     {
//         // Get input parameters as text
//         text *key_arg = PG_GETARG_TEXT_PP(0);
//         text *value_arg = PG_GETARG_TEXT_PP(1);
        
//         // Convert text to C strings safely
//         char *key = text_to_cstring(key_arg);
//         char *value = text_to_cstring(value_arg);
        
//         // Check if cache is full
//         if (current_entries >= cache_capacity) {
//             elog(WARNING, "Cache is full, cannot add more entries");
//             PG_RETURN_BOOL(false);
//         }
        
//         // Find an empty slot or existing key
//         bool entry_added = false;
//         for (int i = 0; i < cache_capacity; i++) {
//             if (cache_entries[i].key == NULL || 
//                 strcmp(cache_entries[i].key, key) == 0) {
                
//                 // Free existing entry if it exists
//                 if (cache_entries[i].key) {
//                     pfree(cache_entries[i].key);
//                     pfree(cache_entries[i].value);
//                 }
                
//                 // Set new entry with careful allocation
//                 cache_entries[i].key = pstrdup(key);
//                 cache_entries[i].value = pstrdup(value);
//                 cache_entries[i].expiry = time(NULL) + 3600;  // 1 hour expiry
                
//                 // Increment entry count if it's a new entry
//                 if (i == current_entries) {
//                     current_entries++;
//                 }
                
//                 entry_added = true;
//                 break;
//             }
//         }
        
//         // Free temporary strings
//         pfree(key);
//         pfree(value);
        
//         // Check if entry was added
//         if (!entry_added) {
//             elog(WARNING, "Could not find a slot to store cache entry");
//             PG_RETURN_BOOL(false);
//         }
//     }
//     PG_CATCH();
//     {
//         // Restore the original memory context
//         MemoryContextSwitchTo(oldcontext);
        
//         // Re-throw the error
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     // Restore the original memory context
//     MemoryContextSwitchTo(oldcontext);
    
//     PG_RETURN_BOOL(true);
// }

// // Get a value from the cache with robust error handling
// PG_FUNCTION_INFO_V1(cache_get);
// Datum cache_get(PG_FUNCTION_ARGS)
// {
//     // Ensure cache is initialized
//     if (cache_entries == NULL) {
//         elog(ERROR, "Cache not initialized. Call cache_init() first.");
//         PG_RETURN_NULL();
//     }

//     // Validate input parameters
//     if (PG_ARGISNULL(0)) {
//         elog(ERROR, "Key cannot be NULL");
//         PG_RETURN_NULL();
//     }

//     // Switch to cache memory context
//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_context);

//     text *result = NULL;
//     PG_TRY();
//     {
//         // Get input key
//         text *key_arg = PG_GETARG_TEXT_PP(0);
//         char *key = text_to_cstring(key_arg);
        
//         // Current time
//         time_t now = time(NULL);
        
//         // Search for the key
//         for (int i = 0; i < current_entries; i++) {
//             if (cache_entries[i].key && 
//                 strcmp(cache_entries[i].key, key) == 0) {
                
//                 // Check if entry has expired
//                 if (now > cache_entries[i].expiry) {
//                     // Remove expired entry
//                     pfree(cache_entries[i].key);
//                     pfree(cache_entries[i].value);
//                     cache_entries[i].key = NULL;
//                     cache_entries[i].value = NULL;
                    
//                     break;
//                 }
                
//                 // Return the value
//                 result = cstring_to_text(cache_entries[i].value);
//                 break;
//             }
//         }
        
//         // Free the key string
//         pfree(key);
//     }
//     PG_CATCH();
//     {
//         // Restore the original memory context
//         MemoryContextSwitchTo(oldcontext);
        
//         // Re-throw the error
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     // Restore the original memory context
//     MemoryContextSwitchTo(oldcontext);
    
//     // Return result (NULL if not found)
//     if (result) {
//         PG_RETURN_TEXT_P(result);
//     }
//     PG_RETURN_NULL();
// }

// // Cleanup function with safe memory management
// // Cleanup function with safe memory management
// PG_FUNCTION_INFO_V1(cache_cleanup);
// Datum cache_cleanup(PG_FUNCTION_ARGS)
// {
//     // Prevent cleanup if no entries exist
//     if (cache_entries == NULL || current_entries == 0) {
//         elog(NOTICE, "No cache entries to clean up");
//         PG_RETURN_BOOL(true);
//     }

//     // Use a top-level memory context for safety
//     MemoryContext oldcontext = CurrentMemoryContext;

//     PG_TRY();
//     {
//         // Explicitly free each entry
//         for (int i = 0; i < current_entries; i++) {
//             if (cache_entries[i].key) {
//                 // Safe pfree with null check
//                 if (cache_entries[i].key != NULL) {
//                     pfree(cache_entries[i].key);
//                     cache_entries[i].key = NULL;
//                 }
                
//                 if (cache_entries[i].value != NULL) {
//                     pfree(cache_entries[i].value);
//                     cache_entries[i].value = NULL;
//                 }
//             }
//         }
        
//         // Free the cache entries array
//         if (cache_entries != NULL) {
//             pfree(cache_entries);
//             cache_entries = NULL;
//         }
        
//         // Reset entry count
//         current_entries = 0;
        
//         // Optional: Log successful cleanup
//         elog(NOTICE, "Cache cleaned up successfully");
//     }
//     PG_CATCH();
//     {
//         // Log any errors during cleanup
//         elog(ERROR, "Error during cache cleanup");
        
//         // Ensure memory is reset even if an error occurs
//         if (cache_entries != NULL) {
//             pfree(cache_entries);
//             cache_entries = NULL;
//         }
//         current_entries = 0;
        
//         // Re-throw the error
//         PG_RE_THROW();
//     }
//     PG_END_TRY();
    
//     PG_RETURN_BOOL(true);
// }

// // Vacuum function to remove expired entries
// PG_FUNCTION_INFO_V1(cache_vacuum);
// Datum cache_vacuum(PG_FUNCTION_ARGS)
// {
//     // Ensure cache is initialized
//     if (cache_entries == NULL) {
//         elog(ERROR, "Cache not initialized. Call cache_init() first.");
//         PG_RETURN_INT32(0);
//     }

//     // Switch to cache memory context
//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_context);

//     int removed_count = 0;
//     PG_TRY();
//     {
//         time_t now = time(NULL);
        
//         for (int i = 0; i < current_entries; i++) {
//             if (cache_entries[i].key && now > cache_entries[i].expiry) {
//                 // Free and nullify expired entry
//                 pfree(cache_entries[i].key);
//                 pfree(cache_entries[i].value);
//                 cache_entries[i].key = NULL;
//                 cache_entries[i].value = NULL;
//                 removed_count++;
//             }
//         }
//     }
//     PG_CATCH();
//     {
//         // Restore the original memory context
//         MemoryContextSwitchTo(oldcontext);
        
//         // Re-throw the error
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     // Restore the original memory context
//     MemoryContextSwitchTo(oldcontext);
    
//     PG_RETURN_INT32(removed_count);
// }






// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "storage/shmem.h"
// #include "utils/memutils.h"
// #include "utils/hsearch.h"
// #include <string.h>
// #include <time.h>

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// // Shared memory key
// #define CACHE_MAGIC_KEY 0x53514C43  // "SQLC" in hex

// // Cache entry structure
// typedef struct {
//     char key[256];   // Fixed-length key to avoid pointer complexities
//     char value[1024]; // Fixed-length value
//     time_t expiry;   // Expiration time
// } CacheEntry;

// // Shared memory structure
// typedef struct {
//     HTAB *cache_table;
//     int cache_capacity;
// } SharedCacheContext;

// // Global shared memory pointer
// static SharedCacheContext *shared_cache_context = NULL;

// // Initialize shared memory
// static void *
// cache_shmem_request(void *request_context)
// {
//     void *shared_memory;
//     Size size;

//     size = MAXALIGN(sizeof(SharedCacheContext));
//     shared_memory = ShmemInitStruct("PostgreSQL_Cache_Extension", 
//                                     size, 
//                                     request_context);
    
//     return shared_memory;
// }

// // Initialization function with shared memory
// PG_FUNCTION_INFO_V1(cache_init);
// Datum cache_init(PG_FUNCTION_ARGS)
// {
//     // Ensure we're in a backend process
//     if (!IsBackendInitialized()) {
//         elog(ERROR, "Cannot initialize cache outside of a backend process");
//         PG_RETURN_BOOL(false);
//     }

//     // Allocate shared memory if not already exists
//     if (shared_cache_context == NULL) {
//         shared_cache_context = (SharedCacheContext *)
//             ShmemInitStruct("PostgreSQL_Cache_Extension", 
//                             sizeof(SharedCacheContext), 
//                             &IsTransactionState);
        
//         if (shared_cache_context == NULL) {
//             elog(ERROR, "Failed to allocate shared memory for cache");
//             PG_RETURN_BOOL(false);
//         }

//         // Initialize hash table configuration
//         HASHCTL info;
//         memset(&info, 0, sizeof(HASHCTL));
//         info.keysize = sizeof(((CacheEntry *)0)->key);
//         info.entrysize = sizeof(CacheEntry);
//         info.max_size = 1000;  // Maximum entries

//         // Create hash table in shared memory
//         shared_cache_context->cache_table = ShmemInitHash(
//             "PostgreSQL_Cache", 
//             100,  // Initial size
//             1000, // Max size
//             &info, 
//             HASH_ELEM | HASH_BLOBS
//         );

//         if (shared_cache_context->cache_table == NULL) {
//             elog(ERROR, "Failed to create shared cache hash table");
//             PG_RETURN_BOOL(false);
//         }

//         // Set default capacity
//         shared_cache_context->cache_capacity = 1000;

//         elog(NOTICE, "Cache initialized successfully");
//     }
    
//     PG_RETURN_BOOL(true);
// }

// // Set a key-value pair in the cache
// PG_FUNCTION_INFO_V1(cache_set);
// Datum cache_set(PG_FUNCTION_ARGS)
// {
//     // Validate input
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) {
//         elog(ERROR, "Key or value cannot be NULL");
//         PG_RETURN_BOOL(false);
//     }

//     // Ensure cache is initialized
//     if (shared_cache_context == NULL || 
//         shared_cache_context->cache_table == NULL) {
//         elog(ERROR, "Cache not initialized. Call cache_init() first.");
//         PG_RETURN_BOOL(false);
//     }

//     // Convert input to C strings
//     text *key_arg = PG_GETARG_TEXT_PP(0);
//     text *value_arg = PG_GETARG_TEXT_PP(1);
    
//     // Safely copy strings with length checks
//     char key[256] = {0};
//     char value[1024] = {0};
    
//     int key_len = Min(VARSIZE_ANY_EXHDR(key_arg), sizeof(key) - 1);
//     int value_len = Min(VARSIZE_ANY_EXHDR(value_arg), sizeof(value) - 1);
    
//     memcpy(key, VARDATA_ANY(key_arg), key_len);
//     memcpy(value, VARDATA_ANY(value_arg), value_len);

//     // Check if entry exists
//     bool found;
//     CacheEntry *entry = hash_search(shared_cache_context->cache_table, 
//                                     key, 
//                                     HASH_ENTER, 
//                                     &found);

//     if (entry) {
//         // Set new values
//         strncpy(entry->key, key, sizeof(entry->key) - 1);
//         strncpy(entry->value, value, sizeof(entry->value) - 1);
//         entry->expiry = time(NULL) + 3600;  // 1 hour expiry
//     } else {
//         elog(WARNING, "Failed to create cache entry");
//         PG_RETURN_BOOL(false);
//     }

//     PG_RETURN_BOOL(true);
// }

// // Get a value from the cache
// PG_FUNCTION_INFO_V1(cache_get);
// Datum cache_get(PG_FUNCTION_ARGS)
// {
//     // Validate input
//     if (PG_ARGISNULL(0)) {
//         elog(ERROR, "Key cannot be NULL");
//         PG_RETURN_NULL();
//     }

//     // Ensure cache is initialized
//     if (shared_cache_context == NULL || 
//         shared_cache_context->cache_table == NULL) {
//         elog(ERROR, "Cache not initialized. Call cache_init() first.");
//         PG_RETURN_NULL();
//     }

//     // Get input key
//     text *key_arg = PG_GETARG_TEXT_PP(0);
    
//     // Safely copy key
//     char key[256] = {0};
//     int key_len = Min(VARSIZE_ANY_EXHDR(key_arg), sizeof(key) - 1);
//     memcpy(key, VARDATA_ANY(key_arg), key_len);
    
//     // Current time
//     time_t now = time(NULL);
    
//     // Lookup entry
//     bool found;
//     CacheEntry *entry = hash_search(shared_cache_context->cache_table, 
//                                     key, 
//                                     HASH_FIND, 
//                                     &found);

//     text *result = NULL;
//     if (found && entry && now <= entry->expiry) {
//         // Found valid entry, convert to text
//         result = cstring_to_text(entry->value);
//     }

//     // Return result (NULL if not found or expired)
//     if (result) {
//         PG_RETURN_TEXT_P(result);
//     }
//     PG_RETURN_NULL();
// }

// // Remove expired entries
// PG_FUNCTION_INFO_V1(cache_vacuum);
// Datum cache_vacuum(PG_FUNCTION_ARGS)
// {
//     // Ensure cache is initialized
//     if (shared_cache_context == NULL || 
//         shared_cache_context->cache_table == NULL) {
//         elog(ERROR, "Cache not initialized. Call cache_init() first.");
//         PG_RETURN_INT32(0);
//     }

//     time_t now = time(NULL);
//     int removed_count = 0;

//     // Iterate through hash table
//     HASH_SEQ_STATUS status;
//     CacheEntry *entry;
//     hash_seq_init(&status, shared_cache_context->cache_table);

//     while ((entry = hash_seq_search(&status)) != NULL) {
//         // Check for expired entries
//         if (now > entry->expiry) {
//             // Remove expired entry
//             bool found;
//             hash_search(shared_cache_context->cache_table, 
//                         entry->key, 
//                         HASH_REMOVE, 
//                         &found);
//             removed_count++;
//         }
//     }

//     PG_RETURN_INT32(removed_count);
// }

// // Complete cache cleanup
// PG_FUNCTION_INFO_V1(cache_cleanup);
// Datum cache_cleanup(PG_FUNCTION_ARGS)
// {
//     // Ensure cache exists
//     if (shared_cache_context == NULL || 
//         shared_cache_context->cache_table == NULL) {
//         elog(NOTICE, "No cache to clean up");
//         PG_RETURN_BOOL(true);
//     }

//     // Destroy hash table
//     hash_destroy(shared_cache_context->cache_table);
//     shared_cache_context->cache_table = NULL;

//     PG_RETURN_BOOL(true);
// }



// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "storage/shmem.h"
// #include "utils/memutils.h"
// #include "utils/hsearch.h"
// #include <string.h>
// #include <time.h>

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// // Cache entry structure
// typedef struct {
//     char *key;       // Key string
//     char *value;     // Value string
//     time_t expiry;   // Expiration timestamp
// } CacheEntry;

// // Global hash table and related structures
// static HTAB *cache_hash_table = NULL;
// static MemoryContext cache_context = NULL;

// // Hash function for the hash table
// static uint32 string_hash(const void *key, Size keysize) {
//     const char *str = (const char *)key;
//     uint32 hash = 5381;
//     int c;

//     while ((c = *str++))
//         hash = ((hash << 5) + hash) + c;

//     return hash;
// }

// // Compare function for hash table keys
// static int string_match(const void *key1, const void *key2, Size keysize) {
//     return strcmp((const char *)key1, (const char *)key2) == 0;
// }

// // Initialize the cache with hash table
// PG_FUNCTION_INFO_V1(cache_init);
// Datum cache_init(PG_FUNCTION_ARGS)
// {
//     // Create a dedicated memory context if not exists
//     if (cache_context == NULL) {
//         cache_context = AllocSetContextCreate(TopMemoryContext,
//                                               "CacheExtensionContext",
//                                               ALLOCSET_SMALL_SIZES);
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_context);

//     PG_TRY();
//     {
//         // Only initialize if not already initialized
//         if (cache_hash_table == NULL) {
//             HASHCTL hash_ctl;
//             MemSet(&hash_ctl, 0, sizeof(HASHCTL));
            
//             hash_ctl.keysize = sizeof(char *);
//             hash_ctl.entrysize = sizeof(CacheEntry);
//             hash_ctl.hash = string_hash;
//             hash_ctl.match = string_match;
//             hash_ctl.hcxt = cache_context;

//             // Create hash table
//             cache_hash_table = hash_create("PostgreSQL Cache",
//                                            100,  // Initial size
//                                            &hash_ctl,
//                                            HASH_ELEM | HASH_CONTEXT | 
//                                            HASH_FUNCTION | HASH_COMPARE);

//             elog(INFO, "Cache initialized with hash table");
//         }
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
//     PG_RETURN_BOOL(true);
// }

// // Set a key-value pair in the cache
// PG_FUNCTION_INFO_V1(cache_set);
// Datum cache_set(PG_FUNCTION_ARGS)
// {
//     // Validate cache initialization
//     if (cache_hash_table == NULL) {
//         elog(ERROR, "Cache not initialized. Call cache_init() first.");
//         PG_RETURN_BOOL(false);
//     }

//     // Validate input parameters
//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) {
//         elog(ERROR, "Key or value cannot be NULL");
//         PG_RETURN_BOOL(false);
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_context);

//     PG_TRY();
//     {
//         // Convert text to C strings
//         text *key_arg = PG_GETARG_TEXT_PP(0);
//         text *value_arg = PG_GETARG_TEXT_PP(1);
        
//         char *key = text_to_cstring(key_arg);
//         char *value = text_to_cstring(value_arg);

//         // Check if key already exists
//         bool found;
//         CacheEntry *entry = (CacheEntry *)hash_search(cache_hash_table, 
//                                                       &key, 
//                                                       HASH_ENTER, 
//                                                       &found);

//         if (found) {
//             // Free existing entry's value if it exists
//             if (entry->value) {
//                 pfree(entry->value);
//             }
//         } else {
//             // New entry, set the key
//             entry->key = pstrdup(key);
//         }

//         // Set value and expiry
//         entry->value = pstrdup(value);
//         entry->expiry = time(NULL) + 3600;  // 1 hour expiry

//         // Free temporary strings
//         pfree(key);
//         pfree(value);
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
//     PG_RETURN_BOOL(true);
// }

// // Get a value from the cache
// PG_FUNCTION_INFO_V1(cache_get);
// Datum cache_get(PG_FUNCTION_ARGS)
// {
//     // Validate cache initialization
//     if (cache_hash_table == NULL) {
//         elog(ERROR, "Cache not initialized. Call cache_init() first.");
//         PG_RETURN_NULL();
//     }

//     // Validate input parameters
//     if (PG_ARGISNULL(0)) {
//         elog(ERROR, "Key cannot be NULL");
//         PG_RETURN_NULL();
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_context);

//     text *result = NULL;
//     PG_TRY();
//     {
//         // Get input key
//         text *key_arg = PG_GETARG_TEXT_PP(0);
//         char *key = text_to_cstring(key_arg);
        
//         time_t now = time(NULL);
        
//         // Search for the entry
//         bool found;
//         CacheEntry *entry = (CacheEntry *)hash_search(cache_hash_table, 
//                                                       &key, 
//                                                       HASH_FIND, 
//                                                       &found);

//         // Check if entry exists and is not expired
//         if (found && now <= entry->expiry) {
//             result = cstring_to_text(entry->value);
//         }

//         // Free the key string
//         pfree(key);
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
    
//     if (result) {
//         PG_RETURN_TEXT_P(result);
//     }
//     PG_RETURN_NULL();
// }

// // Vacuum function to remove expired entries
// PG_FUNCTION_INFO_V1(cache_vacuum);
// Datum cache_vacuum(PG_FUNCTION_ARGS)
// {
//     // Validate cache initialization
//     if (cache_hash_table == NULL) {
//         elog(ERROR, "Cache not initialized. Call cache_init() first.");
//         PG_RETURN_INT32(0);
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_context);

//     int removed_count = 0;
//     PG_TRY();
//     {
//         HASH_SEQ_STATUS status;
//         CacheEntry *entry;
//         time_t now = time(NULL);
        
//         hash_seq_init(&status, cache_hash_table);
        
//         while ((entry = (CacheEntry *)hash_seq_search(&status)) != NULL) {
//             if (now > entry->expiry) {
//                 // Remove expired entry
//                 hash_search(cache_hash_table, &entry->key, HASH_REMOVE, NULL);
//                 removed_count++;
//             }
//         }
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
    
//     PG_RETURN_INT32(removed_count);
// }

// // Cleanup function
// PG_FUNCTION_INFO_V1(cache_cleanup);
// Datum cache_cleanup(PG_FUNCTION_ARGS)
// {
//     // Check if hash table exists
//     if (cache_hash_table == NULL) {
//         elog(NOTICE, "No cache entries to clean up");
//         PG_RETURN_BOOL(true);
//     }

//     MemoryContext oldcontext = CurrentMemoryContext;

//     PG_TRY();
//     {
//         // Destroy the hash table
//         hash_destroy(cache_hash_table);
//         cache_hash_table = NULL;

//         // Optionally destroy the memory context
//         if (cache_context) {
//             MemoryContextDelete(cache_context);
//             cache_context = NULL;
//         }

//         elog(NOTICE, "Cache cleaned up successfully");
//     }
//     PG_CATCH();
//     {
//         elog(ERROR, "Error during cache cleanup");
//         PG_RE_THROW();
//     }
//     PG_END_TRY();
    
//     PG_RETURN_BOOL(true);
// }



// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "storage/shmem.h"
// #include "utils/memutils.h"
// #include "utils/hsearch.h"
// #include <string.h>
// #include <time.h>

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// // Improved cache entry structure with compact memory layout
// typedef struct {
//     char *key;           // Key string 
//     char *value;         // Value string
//     uint64_t expiry;     // Use 64-bit timestamp for extended range
//     bool is_active;      // Track entry status efficiently
// } OptimizedCacheEntry;

// // Enhanced configuration parameters
// #define CACHE_INITIAL_SIZE 1024     // Larger initial size
// #define CACHE_MAX_SIZE 100000       // Prevent unbounded growth
// #define DEFAULT_EXPIRY_TIME 3600    // 1 hour default expiry
// #define LOAD_FACTOR 0.75             // Resize hash table when 75% full

// // Global hash table and related structures
// static HTAB *cache_hash_table = NULL;
// static MemoryContext cache_context = NULL;
// static int current_cache_entries = 0;

// // Advanced string hash function (djb2 algorithm)
// static uint32 advanced_string_hash(const void *key, Size keysize) {
//     const char *str = (const char *)key;
//     uint32 hash = 5381;
//     int c;

//     while ((c = *str++))
//         hash = ((hash << 5) + hash) + c;

//     return hash;
// }

// // Optimized string comparison function
// static int advanced_string_match(const void *key1, const void *key2, Size keysize) {
//     return strcmp((const char *)key1, (const char *)key2) == 0;
// }

// // Resize hash table dynamically
// static bool resize_cache_table() {
//     if (current_cache_entries >= CACHE_MAX_SIZE) {
//         elog(WARNING, "Cache size limit reached. Cannot add more entries.");
//         return false;
//     }

//     // Double the size if load factor exceeds threshold
//     HASHCTL hash_ctl;
//     MemSet(&hash_ctl, 0, sizeof(HASHCTL));
    
//     hash_ctl.keysize = sizeof(char *);
//     hash_ctl.entrysize = sizeof(OptimizedCacheEntry);
//     hash_ctl.hash = advanced_string_hash;
//     hash_ctl.match = advanced_string_match;
//     hash_ctl.hcxt = cache_context;

//     HTAB *new_hash_table = hash_create("Optimized PostgreSQL Cache",
//                                         current_cache_entries * 2,  // Dynamic resizing
//                                         &hash_ctl,
//                                         HASH_ELEM | HASH_CONTEXT | 
//                                         HASH_FUNCTION | HASH_COMPARE);

//     if (new_hash_table == NULL) {
//         elog(ERROR, "Failed to resize cache hash table");
//         return false;
//     }

//     // Optional: Copy existing entries to new table
//     if (cache_hash_table) {
//         HASH_SEQ_STATUS status;
//         OptimizedCacheEntry *entry;
        
//         hash_seq_init(&status, cache_hash_table);
//         while ((entry = (OptimizedCacheEntry *)hash_seq_search(&status)) != NULL) {
//             bool found;
//             OptimizedCacheEntry *new_entry = (OptimizedCacheEntry *)hash_search(
//                 new_hash_table, 
//                 &entry->key, 
//                 HASH_ENTER, 
//                 &found
//             );

//             if (!found) {
//                 new_entry->key = entry->key;
//                 new_entry->value = entry->value;
//                 new_entry->expiry = entry->expiry;
//                 new_entry->is_active = entry->is_active;
//             }
//         }

//         // Destroy old hash table
//         hash_destroy(cache_hash_table);
//     }

//     cache_hash_table = new_hash_table;
//     return true;
// }
// PG_FUNCTION_INFO_V1(cache_init);
// Datum cache_init(PG_FUNCTION_ARGS)
// {
//     // Create a dedicated memory context with better allocation strategy
//     if (cache_context == NULL) {
//         cache_context = AllocSetContextCreate(TopMemoryContext,
//                                               "OptimizedCacheContext",
//                                               ALLOCSET_DEFAULT_SIZES);
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_context);

//     PG_TRY();
//     {
//         // Only create hash table if it doesn't exist
//         if (cache_hash_table == NULL) {
//             HASHCTL hash_ctl;
//             MemSet(&hash_ctl, 0, sizeof(HASHCTL));
            
//             hash_ctl.keysize = sizeof(char *);
//             hash_ctl.entrysize = sizeof(OptimizedCacheEntry);
//             hash_ctl.hash = advanced_string_hash;
//             hash_ctl.match = advanced_string_match;
//             hash_ctl.hcxt = cache_context;

//             // Create hash table with larger initial size
//             cache_hash_table = hash_create("Optimized PostgreSQL Cache",
//                                             CACHE_INITIAL_SIZE,
//                                             &hash_ctl,
//                                             HASH_ELEM | HASH_CONTEXT | 
//                                             HASH_FUNCTION | HASH_COMPARE);

//             elog(INFO, "Optimized cache initialized");
//         }
//         else {
//             // If hash table already exists, just log a notice
//             elog(NOTICE, "Cache already initialized");
//         }
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
//     PG_RETURN_BOOL(true);
// }
// // Optimized cache set function with improved error handling
// PG_FUNCTION_INFO_V1(cache_set);
// Datum cache_set(PG_FUNCTION_ARGS)
// {
//     // Validate cache initialization and input
//     if (cache_hash_table == NULL) {
//         elog(ERROR, "Cache not initialized. Call cache_init() first.");
//         PG_RETURN_BOOL(false);
//     }

//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) {
//         elog(ERROR, "Key or value cannot be NULL");
//         PG_RETURN_BOOL(false);
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_context);

//     PG_TRY();
//     {
//         // Convert text to C strings
//         text *key_arg = PG_GETARG_TEXT_PP(0);
//         text *value_arg = PG_GETARG_TEXT_PP(1);
        
//         char *key = text_to_cstring(key_arg);
//         char *value = text_to_cstring(value_arg);

//         // Resize if needed
//         if (current_cache_entries >= CACHE_INITIAL_SIZE * LOAD_FACTOR) {
//             if (!resize_cache_table()) {
//                 pfree(key);
//                 pfree(value);
//                 PG_RETURN_BOOL(false);
//             }
//         }

//         // Prepare key for hash search
//         char **search_key = palloc(sizeof(char *));
//         *search_key = key;

//         // Check if key already exists
//         bool found;
//         OptimizedCacheEntry *entry = (OptimizedCacheEntry *)hash_search(
//             cache_hash_table, 
//             search_key, 
//             HASH_ENTER, 
//             &found
//         );

//         // Free the temporary search key
//         pfree(search_key);

//         if (found) {
//             // Free existing entry's value
//             if (entry->value) {
//                 pfree(entry->value);
//             }
//         } else {
//             // New entry, set the key and increment counter
//             entry->key = pstrdup(key);
//             current_cache_entries++;
//         }

//         // Set value, expiry, and mark as active
//         entry->value = pstrdup(value);
//         entry->expiry = time(NULL) + DEFAULT_EXPIRY_TIME;
//         entry->is_active = true;

//         // Free temporary strings
//         pfree(key);
//         pfree(value);
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
//     PG_RETURN_BOOL(true);
// }

// PG_FUNCTION_INFO_V1(cache_get);
// Datum cache_get(PG_FUNCTION_ARGS)
// {
//     if (cache_hash_table == NULL) {
//         elog(ERROR, "Cache not initialized. Call cache_init() first.");
//         PG_RETURN_NULL();
//     }

//     if (PG_ARGISNULL(0)) {
//         elog(ERROR, "Key cannot be NULL");
//         PG_RETURN_NULL();
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_context);

//     text *result = NULL;
//     PG_TRY();
//     {
//         text *key_arg = PG_GETARG_TEXT_PP(0);
//         char *key = text_to_cstring(key_arg);
        
//         // Prepare key for hash search
//         char **search_key = palloc(sizeof(char *));
//         *search_key = key;
        
//         time_t now = time(NULL);
        
//         // Search for the entry
//         bool found;
//         OptimizedCacheEntry *entry = (OptimizedCacheEntry *)hash_search(
//             cache_hash_table, 
//             search_key, 
//             HASH_FIND, 
//             &found
//         );

//         // Free the temporary search key
//         pfree(search_key);

//         // More robust checking: entry exists, active, and not expired
//         if (found && entry->is_active && now <= entry->expiry) {
//             result = cstring_to_text(entry->value);
//         }

//         pfree(key);
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
    
//     if (result) {
//         PG_RETURN_TEXT_P(result);
//     }
//     PG_RETURN_NULL();
// }

// // Enhanced vacuum function with more efficient cleanup
// PG_FUNCTION_INFO_V1(cache_vacuum);
// Datum cache_vacuum(PG_FUNCTION_ARGS)
// {
//     if (cache_hash_table == NULL) {
//         elog(ERROR, "Cache not initialized. Call cache_init() first.");
//         PG_RETURN_INT32(0);
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_context);

//     int removed_count = 0;
//     PG_TRY();
//     {
//         HASH_SEQ_STATUS status;
//         OptimizedCacheEntry *entry;
//         time_t now = time(NULL);
        
//         hash_seq_init(&status, cache_hash_table);
        
//         while ((entry = (OptimizedCacheEntry *)hash_seq_search(&status)) != NULL) {
//             // More comprehensive removal criteria
//             if (!entry->is_active || now > entry->expiry) {
//                 hash_search(cache_hash_table, &entry->key, HASH_REMOVE, NULL);
//                 current_cache_entries--;
//                 removed_count++;
//             }
//         }

//         // Optional: Resize down if too many entries are removed
//         if (current_cache_entries < CACHE_INITIAL_SIZE / 2) {
//             resize_cache_table();
//         }
//     }
//     PG_CATCH();
//     {
//         MemoryContextSwitchTo(oldcontext);
//         PG_RE_THROW();
//     }
//     PG_END_TRY();

//     MemoryContextSwitchTo(oldcontext);
    
//     PG_RETURN_INT32(removed_count);
// }

// // Comprehensive cleanup function
// PG_FUNCTION_INFO_V1(cache_cleanup);
// Datum cache_cleanup(PG_FUNCTION_ARGS)
// {
//     if (cache_hash_table == NULL) {
//         elog(NOTICE, "No cache entries to clean up");
//         PG_RETURN_BOOL(true);
//     }

//     MemoryContext oldcontext = CurrentMemoryContext;

//     PG_TRY();
//     {
//         // Destroy the hash table
//         hash_destroy(cache_hash_table);
//         cache_hash_table = NULL;

//         // Reset cache entry counter
//         current_cache_entries = 0;

//         // Delete memory context
//         if (cache_context) {
//             MemoryContextDelete(cache_context);
//             cache_context = NULL;
//         }

//         elog(NOTICE, "Optimized cache cleaned up successfully");
//     }
//     PG_CATCH();
//     {
//         elog(ERROR, "Error during optimized cache cleanup");
//         PG_RE_THROW();
//     }
//     PG_END_TRY();
    
//     PG_RETURN_BOOL(true);
// }


// #include "postgres.h"
// #include "fmgr.h"
// #include "utils/builtins.h"
// #include "storage/shmem.h"
// #include "utils/memutils.h"
// #include "utils/hsearch.h"
// #include <string.h>
// #include <time.h>

// #ifdef PG_MODULE_MAGIC
// PG_MODULE_MAGIC;
// #endif

// // Advanced cache entry with optimizations
// typedef struct {
//     char *key;           // Key string 
//     char *value;         // Value string
//     size_t key_hash;     // Precomputed hash for faster lookups
//     uint64_t timestamp;  // Timestamp for potential advanced caching strategies
// } OptimizedCacheEntry;

// // Configuration parameters
// #define CACHE_INITIAL_SIZE 1024
// #define CACHE_MAX_SIZE 100000
// #define LOAD_FACTOR 0.75

// // Global cache structures
// static HTAB *cache_hash_table = NULL;
// static MemoryContext cache_context = NULL;
// static int current_cache_entries = 0;

// // Advanced string hash function (djb2 algorithm)
// static uint32 advanced_string_hash(const char *str) {
//     uint32 hash = 5381;
//     int c;

//     while ((c = *str++))
//         hash = ((hash << 5) + hash) + c;

//     return hash;
// }

// static bool resize_cache_table() {
//     if (current_cache_entries >= CACHE_MAX_SIZE) {
//         elog(WARNING, "Cache size limit reached");
//         return false;
//     }

//     HASHCTL hash_ctl;
//     MemSet(&hash_ctl, 0, sizeof(HASHCTL));
    
//     hash_ctl.keysize = sizeof(char *);
//     hash_ctl.entrysize = sizeof(OptimizedCacheEntry);
//     hash_ctl.hcxt = cache_context;

//     HTAB *new_hash_table = hash_create("High-Performance PostgreSQL Cache",
//                                         current_cache_entries * 2,
//                                         &hash_ctl,
//                                         HASH_ELEM | HASH_CONTEXT);

//     if (new_hash_table == NULL) {
//         elog(ERROR, "Failed to resize cache hash table");
//         return false;
//     }

//     // Copy existing entries to new table with deep copy
//     if (cache_hash_table) {
//         HASH_SEQ_STATUS status;
//         OptimizedCacheEntry *entry;
        
//         hash_seq_init(&status, cache_hash_table);
//         while ((entry = (OptimizedCacheEntry *)hash_seq_search(&status)) != NULL) {
//             bool found;
//             OptimizedCacheEntry *new_entry = (OptimizedCacheEntry *)hash_search(
//                 new_hash_table, 
//                 &entry->key, 
//                 HASH_ENTER, 
//                 &found
//             );

//             if (!found) {
//                 // Deep copy key and value
//                 new_entry->key = pstrdup(entry->key);
//                 new_entry->value = pstrdup(entry->value);
//                 new_entry->key_hash = entry->key_hash;
//                 new_entry->timestamp = entry->timestamp;
//             }
//         }

//         // Destroy old hash table after successful copy
//         hash_destroy(cache_hash_table);
//     }

//     cache_hash_table = new_hash_table;
//     return true;
// }
// // Initialize cache
// PG_FUNCTION_INFO_V1(cache_init);
// Datum cache_init(PG_FUNCTION_ARGS)
// {
//     // Create a dedicated memory context with optimized allocation
//     if (cache_context == NULL) {
//         cache_context = AllocSetContextCreate(TopMemoryContext,
//                                               "HighPerformanceCacheContext",
//                                               ALLOCSET_DEFAULT_SIZES);
//     }

//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_context);

//     if (cache_hash_table == NULL) {
//         HASHCTL hash_ctl;
//         MemSet(&hash_ctl, 0, sizeof(HASHCTL));
        
//         hash_ctl.keysize = sizeof(char *);
//         hash_ctl.entrysize = sizeof(OptimizedCacheEntry);
//         hash_ctl.hcxt = cache_context;

//         cache_hash_table = hash_create("High-Performance PostgreSQL Cache",
//                                         CACHE_INITIAL_SIZE,
//                                         &hash_ctl,
//                                         HASH_ELEM | HASH_CONTEXT);

//         elog(INFO, "High-performance cache initialized");
//     }

//     MemoryContextSwitchTo(oldcontext);
//     PG_RETURN_BOOL(true);
// }

// // Set value in cache with advanced optimizations
// PG_FUNCTION_INFO_V1(cache_set);
// Datum cache_set(PG_FUNCTION_ARGS)
// {
//     // Validate cache and inputs
//     if (cache_hash_table == NULL) {
//         elog(ERROR, "Cache not initialized");
//         PG_RETURN_BOOL(false);
//     }

//     if (PG_ARGISNULL(0) || PG_ARGISNULL(1)) {
//         elog(ERROR, "Key or value cannot be NULL");
//         PG_RETURN_BOOL(false);
//     }

//     // Switch to cache memory context
//     MemoryContext oldcontext = MemoryContextSwitchTo(cache_context);

//     // Resize if approaching capacity
//     if (current_cache_entries >= CACHE_INITIAL_SIZE * LOAD_FACTOR) {
//         resize_cache_table();
//     }

//     // Convert inputs to C strings
//     text *key_text = PG_GETARG_TEXT_PP(0);
//     text *value_text = PG_GETARG_TEXT_PP(1);
    
//     char *key = text_to_cstring(key_text);
//     char *value = text_to_cstring(value_text);

//     // Compute hash for faster future lookups
//     size_t key_hash = advanced_string_hash(key);

//     // Search for existing entry or create new one
//     bool found;
//     OptimizedCacheEntry *entry = (OptimizedCacheEntry *)hash_search(
//         cache_hash_table, 
//         &key,  
//         HASH_ENTER, 
//         &found
//     );

//     // Update or create entry
//     if (found) {
//         // Free existing value if present
//         if (entry->value) {
//             pfree(entry->value);
//         }
//     } else {
//         // New entry
//         entry->key = pstrdup(key);
//         current_cache_entries++;
//     }

//     // Set new value and metadata
//     entry->value = pstrdup(value);
//     entry->key_hash = key_hash;
//     entry->timestamp = (uint64_t)time(NULL);

//     // Clean up
//     pfree(key);
//     pfree(value);

//     MemoryContextSwitchTo(oldcontext);
//     PG_RETURN_BOOL(true);
// }

// // Get value from cache with optimized retrieval
// PG_FUNCTION_INFO_V1(cache_get);
// Datum cache_get(PG_FUNCTION_ARGS)
// {
//     if (cache_hash_table == NULL) {
//         elog(ERROR, "Cache not initialized");
//         PG_RETURN_NULL();
//     }

//     // Convert input to C string
//     text *key_text = PG_GETARG_TEXT_PP(0);
//     char *key = text_to_cstring(key_text);

//     // Compute hash for faster lookup
//     size_t key_hash = advanced_string_hash(key);

//     // Search for entry
//     bool found;
//     OptimizedCacheEntry *entry = (OptimizedCacheEntry *)hash_search(
//         cache_hash_table, 
//         &key,  
//         HASH_FIND, 
//         &found
//     );

//     // Free temporary key string
//     pfree(key);

//     // Return value if found
//     if (found && entry->value) {
//         text *result = cstring_to_text(entry->value);
//         PG_RETURN_TEXT_P(result);
//     }

//     PG_RETURN_NULL();
// }

// // Optional: Add a cache cleanup function
// PG_FUNCTION_INFO_V1(cache_cleanup);
// Datum cache_cleanup(PG_FUNCTION_ARGS)
// {
//     if (cache_hash_table) {
//         hash_destroy(cache_hash_table);
//         cache_hash_table = NULL;
//     }

//     if (cache_context) {
//         MemoryContextDelete(cache_context);
//         cache_context = NULL;
//     }

//     current_cache_entries = 0;
//     PG_RETURN_BOOL(true);
// }