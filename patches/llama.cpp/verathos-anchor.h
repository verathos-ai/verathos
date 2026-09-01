// Verathos streaming execution-anchor primitives (portable host C).
//
// Byte-compatible with the Python side:
//   zkllm.crypto.merkle.hash_leaf  = blake3("VERILLM_LEAF_V1" + data)
//   zkllm.crypto.merkle.hash_node  = blake3("VERILLM_NODE_V1" + l + r)
//   verallm.mesh.execution_anchor.execution_anchor_row_leaf_hash_v3:
//     pad row to a 2048-byte lane multiple, hash_leaf() each lane,
//     odd-duplicate Merkle-reduce the lane hashes to one 32-byte outer
//     leaf; outer leaves fold into an O(log n) Merkle mountain-range
//     frontier whose bagged root is byte-identical to MerkleTree over the
//     same leaves.
//
// Deliberately dependency-free single-header C (works in .c, .cpp, .cu
// translation units) so the llama.cpp patch can hash rows on ANY backend
// host-side: CUDA boxes, Metal/Apple-silicon (unified memory), plain CPU.
// The compression math is transliterated from zkllm/cuda/blake3_merkle.cu,
// whose output is already validated against the Python blake3 package.

#ifndef VERATHOS_ANCHOR_H
#define VERATHOS_ANCHOR_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VERATHOS_ANCHOR_LANE_BYTES 2048
#define VERATHOS_ANCHOR_HASH_BYTES 32
// 32 levels bounds the frontier at 2^32-1 rows, matching the Python port.
#define VERATHOS_ANCHOR_MAX_LEVELS 32

// ----------------------------------------------------------------------
// BLAKE3 core
// ----------------------------------------------------------------------

static const uint32_t VERATHOS_B3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

static const int VERATHOS_B3_SCHEDULE[7][16] = {
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
    { 2,  6,  3, 10,  7,  0,  4, 13,  1, 11, 12,  5,  9, 14, 15,  8},
    { 3,  4, 10, 12, 13,  2,  7, 14,  6,  5,  9,  0, 11, 15,  8,  1},
    {10,  7, 12,  9, 14,  3, 13, 15,  4,  0, 11,  2,  5,  8,  1,  6},
    {12, 13,  9, 11, 15, 10, 14,  8,  7,  2,  5,  3,  0,  1,  6,  4},
    { 9, 14, 11,  5,  8, 12, 15,  1, 13,  3,  0, 10,  2,  6,  4,  7},
    {11, 15,  5,  0,  1,  9,  8,  6, 14, 10,  2, 12,  3,  4,  7, 13},
};

#define VERATHOS_B3_CHUNK_START 1u
#define VERATHOS_B3_CHUNK_END   2u
#define VERATHOS_B3_PARENT      4u
#define VERATHOS_B3_ROOT        8u

static inline uint32_t verathos_b3_rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static inline uint32_t verathos_b3_load_le32(const uint8_t *p) {
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

static inline void verathos_b3_store_le32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

#define VERATHOS_B3_G(a, b, c, d, mx, my)                     \
    do {                                                      \
        a = a + b + (mx); d = verathos_b3_rotr32(d ^ a, 16);  \
        c = c + d;        b = verathos_b3_rotr32(b ^ c, 12);  \
        a = a + b + (my); d = verathos_b3_rotr32(d ^ a, 8);   \
        c = c + d;        b = verathos_b3_rotr32(b ^ c, 7);   \
    } while (0)

static void verathos_b3_compress(
    uint32_t cv[8],
    const uint32_t block[16],
    uint32_t counter_lo,
    uint32_t counter_hi,
    uint32_t block_len,
    uint32_t flags)
{
    uint32_t s[16];
    s[0]  = cv[0]; s[1] = cv[1]; s[2] = cv[2]; s[3] = cv[3];
    s[4]  = cv[4]; s[5] = cv[5]; s[6] = cv[6]; s[7] = cv[7];
    s[8]  = VERATHOS_B3_IV[0]; s[9]  = VERATHOS_B3_IV[1];
    s[10] = VERATHOS_B3_IV[2]; s[11] = VERATHOS_B3_IV[3];
    s[12] = counter_lo; s[13] = counter_hi;
    s[14] = block_len;  s[15] = flags;
    for (int r = 0; r < 7; r++) {
        const int *m = VERATHOS_B3_SCHEDULE[r];
        VERATHOS_B3_G(s[0], s[4], s[8],  s[12], block[m[0]],  block[m[1]]);
        VERATHOS_B3_G(s[1], s[5], s[9],  s[13], block[m[2]],  block[m[3]]);
        VERATHOS_B3_G(s[2], s[6], s[10], s[14], block[m[4]],  block[m[5]]);
        VERATHOS_B3_G(s[3], s[7], s[11], s[15], block[m[6]],  block[m[7]]);
        VERATHOS_B3_G(s[0], s[5], s[10], s[15], block[m[8]],  block[m[9]]);
        VERATHOS_B3_G(s[1], s[6], s[11], s[12], block[m[10]], block[m[11]]);
        VERATHOS_B3_G(s[2], s[7], s[8],  s[13], block[m[12]], block[m[13]]);
        VERATHOS_B3_G(s[3], s[4], s[9],  s[14], block[m[14]], block[m[15]]);
    }
    for (int i = 0; i < 8; i++) {
        cv[i] = s[i] ^ s[i + 8];
    }
}

// One BLAKE3 chunk (<= 1024 bytes) at the given chunk counter. extra_flags
// is OR-ed into every block (VERATHOS_B3_ROOT for a single-chunk message).
static void verathos_b3_hash_chunk(
    const uint8_t *data, uint32_t data_len, uint64_t counter,
    uint32_t extra_flags, uint8_t cv_out[32])
{
    uint32_t num_blocks = (data_len + 63u) / 64u;
    if (num_blocks == 0) num_blocks = 1;
    uint32_t cv[8];
    for (int i = 0; i < 8; i++) cv[i] = VERATHOS_B3_IV[i];
    for (uint32_t b = 0; b < num_blocks; b++) {
        uint32_t block_words[16];
        uint32_t block_start = b * 64u;
        for (int w = 0; w < 16; w++) {
            uint32_t off = block_start + (uint32_t)w * 4u;
            uint8_t b0 = (off     < data_len) ? data[off]     : 0;
            uint8_t b1 = (off + 1 < data_len) ? data[off + 1] : 0;
            uint8_t b2 = (off + 2 < data_len) ? data[off + 2] : 0;
            uint8_t b3 = (off + 3 < data_len) ? data[off + 3] : 0;
            block_words[w] = (uint32_t)b0
                           | ((uint32_t)b1 << 8)
                           | ((uint32_t)b2 << 16)
                           | ((uint32_t)b3 << 24);
        }
        uint32_t flags = 0;
        if (b == 0)              flags |= VERATHOS_B3_CHUNK_START;
        if (b == num_blocks - 1) flags |= VERATHOS_B3_CHUNK_END | extra_flags;
        uint32_t block_len = (b == num_blocks - 1)
                           ? (data_len - b * 64u)
                           : 64u;
        if (data_len == 0) block_len = 0;
        verathos_b3_compress(cv, block_words,
                             (uint32_t)(counter & 0xFFFFFFFFu),
                             (uint32_t)(counter >> 32),
                             block_len, flags);
    }
    for (int i = 0; i < 8; i++) verathos_b3_store_le32(cv_out + i * 4, cv[i]);
}

static void verathos_b3_parent(
    const uint8_t left_cv[32], const uint8_t right_cv[32],
    int is_root, uint8_t out[32])
{
    uint32_t block_words[16];
    for (int i = 0; i < 8; i++) {
        block_words[i]     = verathos_b3_load_le32(left_cv + i * 4);
        block_words[i + 8] = verathos_b3_load_le32(right_cv + i * 4);
    }
    uint32_t cv[8];
    for (int i = 0; i < 8; i++) cv[i] = VERATHOS_B3_IV[i];
    uint32_t flags = VERATHOS_B3_PARENT | (is_root ? VERATHOS_B3_ROOT : 0u);
    verathos_b3_compress(cv, block_words, 0, 0, 64, flags);
    for (int i = 0; i < 8; i++) verathos_b3_store_le32(out + i * 4, cv[i]);
}

// General BLAKE3 for messages up to a bounded chunk count (chunk merge
// stack, ROOT on the final merge only). Domain-prefixed messages here are
// at most 15 + 2048 = 2063 bytes = 3 chunks, but the loop is general.
static void verathos_b3_hash(
    const uint8_t *msg, uint32_t msg_len, uint8_t out[32])
{
    uint32_t num_chunks = (msg_len + 1023u) / 1024u;
    if (num_chunks <= 1) {
        verathos_b3_hash_chunk(msg, msg_len, 0, VERATHOS_B3_ROOT, out);
        return;
    }
    // Chaining-value merge stack, binary-counter style, for every chunk but
    // the last: after pushing chunk i (0-based), merge while the pushed
    // count has trailing zero bits. The FINAL chunk never eager-merges;
    // it starts the bagging pass that folds the stack right-to-left, with
    // ROOT set only on the very last parent, matching the BLAKE3 tree rule
    // that the left subtree holds the largest power of two of the chunks.
    uint8_t stack[54][32];
    uint32_t stack_len = 0;
    for (uint32_t chunk = 0; chunk + 1 < num_chunks; chunk++) {
        uint32_t offset = chunk * 1024u;
        uint8_t cv[32];
        verathos_b3_hash_chunk(msg + offset, 1024u, chunk, 0, cv);
        uint32_t total = chunk + 1;
        while ((total & 1u) == 0u) {
            verathos_b3_parent(stack[stack_len - 1], cv, 0, cv);
            stack_len--;
            total >>= 1;
        }
        memcpy(stack[stack_len], cv, 32);
        stack_len++;
    }
    uint8_t cv[32];
    {
        uint32_t offset = (num_chunks - 1) * 1024u;
        verathos_b3_hash_chunk(
            msg + offset, msg_len - offset, num_chunks - 1, 0, cv);
    }
    while (stack_len > 0) {
        int is_root = (stack_len == 1);
        verathos_b3_parent(stack[stack_len - 1], cv, is_root, cv);
        stack_len--;
    }
    memcpy(out, cv, 32);
}

// ----------------------------------------------------------------------
// Domain-separated leaf/node hashing (zkllm.crypto.merkle equivalents)
// ----------------------------------------------------------------------

// hash_leaf(data) = blake3("VERILLM_LEAF_V1" + data).
// Copies the domain prefix and payload into one buffer so chunk boundaries
// land exactly where the Python side puts them. Callers pass lane-sized
// payloads (<= 2048 bytes).
static void verathos_hash_leaf(
    const uint8_t *data, uint32_t data_len, uint8_t out[32])
{
    static const uint8_t domain[15] = {
        'V', 'E', 'R', 'I', 'L', 'L', 'M', '_',
        'L', 'E', 'A', 'F', '_', 'V', '1'
    };
    uint8_t buffer[15 + VERATHOS_ANCHOR_LANE_BYTES];
    memcpy(buffer, domain, sizeof(domain));
    memcpy(buffer + sizeof(domain), data, data_len);
    verathos_b3_hash(buffer, (uint32_t)(sizeof(domain) + data_len), out);
}

// hash_node(left, right) = blake3("VERILLM_NODE_V1" + left + right).
static void verathos_hash_node(
    const uint8_t left[32], const uint8_t right[32], uint8_t out[32])
{
    static const uint8_t domain[15] = {
        'V', 'E', 'R', 'I', 'L', 'L', 'M', '_',
        'N', 'O', 'D', 'E', '_', 'V', '1'
    };
    uint8_t buffer[15 + 64];
    memcpy(buffer, domain, sizeof(domain));
    memcpy(buffer + sizeof(domain), left, 32);
    memcpy(buffer + sizeof(domain) + 32, right, 32);
    verathos_b3_hash(buffer, sizeof(buffer), out);
}

// ----------------------------------------------------------------------
// Row leaf: 2048-byte lanes, odd-duplicate reduce (execution_anchor_v3)
// ----------------------------------------------------------------------

// execution_anchor_row_leaf_hash_v3 equivalent: the row is padded with
// zeros to a lane multiple, each lane goes through hash_leaf, and the lane
// hashes reduce with hash_node duplicating a trailing odd element.
static void verathos_anchor_row_leaf(
    const uint8_t *row_bytes, uint32_t row_width, uint8_t out[32])
{
    // Bounded scratch: enough lane hashes for the widest committed row
    // (an LM-head f32 logits row at 152k vocab is 297 lanes; 1024 lanes
    // covers 2 MiB rows).
    enum { VERATHOS_ANCHOR_MAX_LANES = 1024 };
    uint8_t level[VERATHOS_ANCHOR_MAX_LANES][32];
    uint32_t lane_count =
        (row_width + VERATHOS_ANCHOR_LANE_BYTES - 1)
        / VERATHOS_ANCHOR_LANE_BYTES;
    if (lane_count == 0 || lane_count > VERATHOS_ANCHOR_MAX_LANES) {
        memset(out, 0, 32);
        return;
    }
    uint8_t lane[VERATHOS_ANCHOR_LANE_BYTES];
    for (uint32_t i = 0; i < lane_count; i++) {
        uint32_t offset = i * VERATHOS_ANCHOR_LANE_BYTES;
        uint32_t len = row_width - offset;
        if (len > VERATHOS_ANCHOR_LANE_BYTES) len = VERATHOS_ANCHOR_LANE_BYTES;
        memcpy(lane, row_bytes + offset, len);
        if (len < VERATHOS_ANCHOR_LANE_BYTES) {
            memset(lane + len, 0, VERATHOS_ANCHOR_LANE_BYTES - len);
        }
        verathos_hash_leaf(lane, VERATHOS_ANCHOR_LANE_BYTES, level[i]);
    }
    uint32_t count = lane_count;
    while (count > 1) {
        uint32_t next = 0;
        for (uint32_t i = 0; i < count; i += 2) {
            const uint8_t *left = level[i];
            const uint8_t *right = (i + 1 < count) ? level[i + 1] : level[i];
            verathos_hash_node(left, right, level[next]);
            next++;
        }
        count = next;
    }
    memcpy(out, level[0], 32);
}

// ----------------------------------------------------------------------
// Canonical stage identifiers
// ----------------------------------------------------------------------

// Builds "<tensor>:<side>" as a canonical anchor stage id.
//
// ggml decorates tensor names with the owning buffer, for example
// "CUDA0#blk.0.ffn_gate.weight#0" on CUDA and a bare
// "blk.0.ffn_gate.weight" on CPU. Committing the decorated name would make
// the same op produce different stage ids on different backends, so the
// wrapper is stripped: with two or more '#', the segment between the first
// and last is the real tensor name. The result is lowercased and reduced to
// the identifier alphabet the verifier accepts, which keeps a CUDA, Metal,
// and CPU serve of the same model byte-identical at the anchor layer.
static void verathos_anchor_stage_id(
    const char *tensor_name, const char *side, char *out, uint32_t out_size)
{
    if (out == NULL || out_size == 0) {
        return;
    }
    out[0] = '\0';
    if (tensor_name == NULL || side == NULL) {
        return;
    }
    const char *begin = tensor_name;
    const char *end = tensor_name + strlen(tensor_name);
    const char *first = strchr(tensor_name, '#');
    if (first != NULL) {
        const char *last = strrchr(tensor_name, '#');
        if (last > first) {
            begin = first + 1;
            end = last;
        }
    }
    uint32_t index = 0;
    for (const char *p = begin; p < end && index + 1 < out_size; p++, index++) {
        char c = *p;
        if (c >= 'A' && c <= 'Z') {
            c = (char)(c - 'A' + 'a');
        }
        const int ok = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                       c == '_' || c == '.' || c == '-' || c == '/';
        out[index] = ok ? c : '_';
    }
    out[index] = '\0';
    if (index == 0 ||
        !((out[0] >= 'a' && out[0] <= 'z') || (out[0] >= '0' && out[0] <= '9'))) {
        // The verifier requires an alphanumeric first character.
        if (index + 2 < out_size) {
            memmove(out + 1, out, index + 1);
            out[0] = 't';
            index++;
        }
    }
    if (index + 1 < out_size) {
        snprintf(out + index, out_size - index, ":%s", side);
    }
}

// ----------------------------------------------------------------------
// Streaming frontier (MMR), byte-identical to StreamingExecutionAnchorV3
// ----------------------------------------------------------------------

typedef struct {
    uint8_t  peaks[VERATHOS_ANCHOR_MAX_LEVELS][32];
    uint8_t  occupied[VERATHOS_ANCHOR_MAX_LEVELS];
    uint32_t count;
} verathos_anchor_frontier;

static void verathos_anchor_frontier_init(verathos_anchor_frontier *f) {
    memset(f, 0, sizeof(*f));
}

static int verathos_anchor_frontier_append(
    verathos_anchor_frontier *f, const uint8_t leaf[32])
{
    if (f->count >= 0xFFFFFFFEu) return 0;
    uint8_t carry[32];
    memcpy(carry, leaf, 32);
    uint32_t level = 0;
    while (level < VERATHOS_ANCHOR_MAX_LEVELS && f->occupied[level]) {
        verathos_hash_node(f->peaks[level], carry, carry);
        f->occupied[level] = 0;
        level++;
    }
    if (level >= VERATHOS_ANCHOR_MAX_LEVELS) return 0;
    memcpy(f->peaks[level], carry, 32);
    f->occupied[level] = 1;
    f->count++;
    return 1;
}

// Bagged root, byte-identical to MerkleTree over the same leaves: odd
// peaks lift with hash_node(carry, carry) until levels align, then join
// left-peak-first exactly like the Python port.
static int verathos_anchor_frontier_root(
    const verathos_anchor_frontier *f, uint8_t out[32])
{
    if (f->count == 0) return 0;
    uint32_t level = 0;
    while (level < VERATHOS_ANCHOR_MAX_LEVELS && !f->occupied[level]) level++;
    if (level >= VERATHOS_ANCHOR_MAX_LEVELS) return 0;
    uint8_t carry[32];
    memcpy(carry, f->peaks[level], 32);
    for (uint32_t next = level + 1; next < VERATHOS_ANCHOR_MAX_LEVELS; next++) {
        if (!f->occupied[next]) continue;
        while (level < next) {
            verathos_hash_node(carry, carry, carry);
            level++;
        }
        verathos_hash_node(f->peaks[next], carry, carry);
        level++;
    }
    memcpy(out, carry, 32);
    return 1;
}

#ifdef __cplusplus
}
#endif

#endif // VERATHOS_ANCHOR_H
