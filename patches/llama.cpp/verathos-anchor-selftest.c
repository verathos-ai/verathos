#include <stdio.h>
#include <stdlib.h>
#include "verathos-anchor.h"

static void print_hex(const char *label, const uint8_t h[32]) {
    printf("%s=", label);
    for (int i = 0; i < 32; i++) printf("%02x", h[i]);
    printf("\n");
}

int main(void) {
    // Deterministic pseudo-random bytes (LCG) shared with the Python side.
    uint64_t state = 0x243F6A8885A308D3ull;
    uint8_t buf[1 << 20];
    for (size_t i = 0; i < sizeof(buf); i++) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        buf[i] = (uint8_t)(state >> 33);
    }

    // 1. Raw blake3 at boundary lengths (0..3 chunks incl. block edges).
    const uint32_t lengths[] = {0, 1, 63, 64, 65, 1023, 1024, 1025,
                                2047, 2048, 2063, 2064, 3000};
    for (unsigned i = 0; i < sizeof(lengths) / sizeof(lengths[0]); i++) {
        uint8_t out[32];
        verathos_b3_hash(buf, lengths[i], out);
        char label[32];
        snprintf(label, sizeof(label), "b3_%u", lengths[i]);
        print_hex(label, out);
    }

    // 2. hash_leaf / hash_node.
    uint8_t leaf[32], node[32];
    verathos_hash_leaf(buf, 2048, leaf);
    print_hex("leaf_2048", leaf);
    verathos_hash_node(buf, buf + 32, node);
    print_hex("node", node);

    // 3. Row leaves at several widths (sub-lane, exact, multi-lane, odd).
    const uint32_t widths[] = {100, 2048, 4096, 5000, 14336, 75776};
    for (unsigned i = 0; i < sizeof(widths) / sizeof(widths[0]); i++) {
        uint8_t out[32];
        verathos_anchor_row_leaf(buf, widths[i], out);
        char label[32];
        snprintf(label, sizeof(label), "rowleaf_%u", widths[i]);
        print_hex(label, out);
    }

    // 4. Stage-id canonicalization across backend name decorations.
    {
        const char *names[] = {
            "CUDA0#blk.0.ffn_gate.weight#0",
            "blk.0.ffn_gate.weight",
            "Metal#blk.0.ffn_gate.weight#3",
            "RPC0[10.0.0.1:50052]#blk.0.ffn_gate.weight#1",
            "#leading",
        };
        for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
            char stage[256];
            verathos_anchor_stage_id(names[i], "dst", stage, sizeof(stage));
            printf("stage_%u=%s\n", i, stage);
        }
    }

    // 5. Frontier roots across count shapes.
    const uint32_t counts[] = {1, 2, 3, 5, 8, 33, 100};
    for (unsigned i = 0; i < sizeof(counts) / sizeof(counts[0]); i++) {
        verathos_anchor_frontier frontier;
        verathos_anchor_frontier_init(&frontier);
        for (uint32_t row = 0; row < counts[i]; row++) {
            uint8_t rl[32];
            verathos_anchor_row_leaf(buf + (row * 977 % 4096), 14336, rl);
            if (!verathos_anchor_frontier_append(&frontier, rl)) return 2;
        }
        uint8_t root[32];
        if (!verathos_anchor_frontier_root(&frontier, root)) return 3;
        char label[32];
        snprintf(label, sizeof(label), "root_%u", counts[i]);
        print_hex(label, root);
    }
    return 0;
}
