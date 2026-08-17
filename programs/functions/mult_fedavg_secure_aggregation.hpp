#pragma once

// "mult_fedavg" secure aggregation variant for Flotilla (see
// flotilla/docs/secure_aggregation/hpmpc_backend.md and
// docs/secure_aggregation/mult_fedavg.md). Unlike fedavg_secure_aggregation.hpp
// (which just sums ALREADY-PRE-WEIGHTED client updates, summed in Python
// before this program ever runs), this variant receives each client's RAW
// (unweighted) update and RAW dataset size separately, and computes
// weight_i * dataset_size_i via genuine secret x secret multiplication
// INSIDE the MPC circuit for every client, then sums those products across
// clients -- so the servers genuinely have to communicate to compute the
// product, not just to combine shares. This is why the file format and
// control flow differ so much from fedavg_secure_aggregation.hpp's "read
// one pre-summed row per element" design; see that file's docstring for
// the contrasting, simpler case.
//
// Like fedavg_secure_aggregation.hpp, this program does NOT reveal its
// final result among the compute parties -- it exports each party's own
// raw share of the summed product (and of the summed dataset size), and
// reconstruction happens entirely at flo_server (see
// server/secure_agg/reconstruct.py). The multiply step (Round 2 below)
// STILL needs real inter-party communication -- that's inherent to secure
// multiplication and unaffected by removing the final reveal.
//
// Currently PROTOCOL=5 (Trio) ONLY. This is a deliberate scope limit, not an
// oversight: the native-fragment-input + prepare_mult composition this file
// depends on (see below) was verified empirically for Trio specifically;
// Replicated (PROTOCOL=2) and Tetrad (PROTOCOL=8) would each need their own
// from-scratch verification of the same composition before being wired in
// here (their prepare_mult/complete_mult implementations, and whether their
// own prepare_receive_from-based native input composes correctly with
// prepare_mult, have not been checked) -- see mult_fedavg.md's "known
// limitations" section.
//
// How multiplication-compatible shares are obtained (the key design
// decision here, and the one that replaced an earlier, failed approach):
// rather than hand-constructing a raw Trio (p1, p2) share pair for a
// client's FULL secret value and feeding it directly into prepare_mult
// (tried during development -- confirmed to give a numerically WRONG
// product, root-caused to prepare_mult's Beaver-triple-style masking not
// composing with an externally-injected raw share the way it does with a
// share produced by hpmpc's own sharing), each of the 3 parties instead
// inputs its OWN already-known replicated3pc share fragment (c_j -- the
// same value it already receives from the client via SubmitShare for the
// existing reveal-only fedavg path) using hpmpc's NATIVE, tested-working
// prepare_receive_from<P_j> mechanism -- exactly what test_basic_primitives.hpp
// (hpmpc's own test suite) uses for ITS multiplication test operands. All 3
// parties run all 3 of these calls (their own real fragment, template-typed
// P_j for the other two roles with an unused placeholder value), batched
// under one communicate(); native Additive_Share::operator+ (verified
// elsewhere in this codebase to be plain, correct elementwise addition for
// Trio -- see fedavg_secure_aggregation.hpp's docstring) then combines the
// 3 fragments into a share that DOES compose correctly with prepare_mult,
// since it was built entirely from hpmpc's own proven input+add primitives
// rather than a hand-rolled raw construction. Verified end to end in Docker
// (real Replicated3PCScheme shares of 3.0 and 5.0, frac_bits=13): revealed
// raw product 1006632960 == 24576*40960 exactly, on all 3 parties.
//
// Why complete_mult_without_trunc(), not complete_mult(): Trio's
// complete_mult_with_trunc() (protocols/3-PC/ours/oecl-P_0_template.hpp and
// siblings -- the truncated-fixed-point completion that Additive_Share::
// complete_mult() dispatches to whenever FRACTIONAL>0) is an EMPTY STUB for
// Trio -- confirmed by reading the function body, and confirmed to silently
// produce a garbage (untouched) share rather than crash. So this program
// always uses complete_mult_without_trunc(), which gives the RAW,
// UNTRUNCATED product (i.e. with 2*frac_bits of fractional precision,
// since both operands are frac_bits-encoded fixed-point values) --
// verified numerically (1006632960 / 2**26 == 15.0 == 3.0*5.0). No in-MPC
// truncation is needed at all: the final product-sum is revealed and
// decoded on the Python side anyway (mirroring the existing
// DATASET_SIZE_LAYER_NAME reveal-and-decode-in-Python pattern), so Python
// just decodes this particular revealed field at 2*frac_bits instead of
// frac_bits -- see backend_hpmpc.py's mult-fedavg code path.
//
// File contract (see backend_hpmpc.py):
//   input file (this party's own fragments ONLY -- see the docstring above
//   for why no other party's data is needed in this file at all):
//     uint32 num_clients
//     uint32 elements_per_client
//     for each of num_clients clients, in a fixed (round-stable) order:
//       elements_per_client x uint64  -- this party's own replicated3pc
//                                         share fragment (c_j) of this
//                                         client's RAW (unweighted) update,
//                                         one per flattened weight element
//       1 x uint64                    -- this party's own replicated3pc
//                                         share fragment (c_j) of this
//                                         client's RAW dataset size
//   output file (this party's own raw Trio share of the result -- NOT
//   revealed; see module docstring):
//     uint32 elements_per_client
//     elements_per_client x (p1, p2) uint64 pairs -- this party's own raw
//                                       Trio share of sum-across-clients of
//                                       weight_i*dataset_size_i. RAW/
//                                       untruncated (2*frac_bits fractional
//                                       precision once reconstructed --
//                                       decode accordingly)
//     1 x (p1, p2) uint64 pair       -- this party's own raw Trio share of
//                                       sum-across-clients of dataset_size_i
//                                       (normal frac_bits once reconstructed
//                                       -- decode like the existing
//                                       DATASET_SIZE_LAYER_NAME field)
//
// Instantiated twice per protocol_executer.hpp's init/live convention (see
// fedavg_secure_aggregation.hpp's docstring for the full explanation) --
// both phases read the SAME num_clients/elements_per_client from the SAME
// input file, so both execute identically-shaped loops issuing identical
// counts of prepare_receive_from/prepare_mult calls (Rounds 1 and 2 below
// still need real communication -- only the former final reveal round was
// removed), which is what keeps the init phase's buffer-size bookkeeping
// consistent with what the live phase actually sends.

#include "../../datatypes/Additive_Share.hpp"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <type_traits>
#include <vector>

#define RESULTTYPE DATATYPE
#define FUNCTION MultFedAvgSecureAggregation

#if PROTOCOL == 5
    #if PARTY == 0
        #define TRIO_LIVE_SHARE OECL0_Share<DATATYPE>
    #elif PARTY == 1
        #define TRIO_LIVE_SHARE OECL1_Share<DATATYPE>
    #elif PARTY == 2
        #define TRIO_LIVE_SHARE OECL2_Share<DATATYPE>
    #endif
#endif

template <typename Share>
void MultFedAvgSecureAggregation(DATATYPE* res)
{
#if PROTOCOL != 5
    print("mult_fedavg_secure_aggregation: only PROTOCOL=5 (Trio) is supported\n");
    exit(1);
#else
    using A = Additive_Share<DATATYPE, Share>;

    const char* input_path = std::getenv("SECURE_AGG_INPUT_FILE");
    const char* output_path = std::getenv("SECURE_AGG_OUTPUT_FILE");
    if (input_path == nullptr || output_path == nullptr)
    {
        print("mult_fedavg_secure_aggregation: SECURE_AGG_INPUT_FILE / SECURE_AGG_OUTPUT_FILE not set\n");
        exit(1);
    }

    uint32_t num_clients = 0;
    uint32_t elements_per_client = 0;
    std::vector<std::vector<uint64_t>> own_weight_frag;  // [client][element]
    std::vector<uint64_t> own_dataset_size_frag;          // [client]

    {
        std::ifstream in(input_path, std::ios::binary);
        if (!in.is_open())
        {
            print("mult_fedavg_secure_aggregation: failed to open input file\n");
            exit(1);
        }
        in.read(reinterpret_cast<char*>(&num_clients), sizeof(num_clients));
        in.read(reinterpret_cast<char*>(&elements_per_client), sizeof(elements_per_client));

        own_weight_frag.assign(num_clients, std::vector<uint64_t>(elements_per_client, 0));
        own_dataset_size_frag.assign(num_clients, 0);
        for (uint32_t c = 0; c < num_clients; c++)
        {
            for (uint32_t k = 0; k < elements_per_client; k++)
                in.read(reinterpret_cast<char*>(&own_weight_frag[c][k]), sizeof(uint64_t));
            in.read(reinterpret_cast<char*>(&own_dataset_size_frag[c]), sizeof(uint64_t));
        }
    }

    // Round 1: every party inputs its OWN known fragment for every client's
    // every weight element, and every client's dataset size, via hpmpc's
    // native prepare_receive_from<P_j> -- one instance per owning role (0,
    // 1, 2), batched under a single communicate(). Non-owning roles pass an
    // unused placeholder (0); see the module docstring for why this is safe.
    std::vector<std::vector<A>> weight_frag_p0(num_clients, std::vector<A>(elements_per_client));
    std::vector<std::vector<A>> weight_frag_p1(num_clients, std::vector<A>(elements_per_client));
    std::vector<std::vector<A>> weight_frag_p2(num_clients, std::vector<A>(elements_per_client));
    std::vector<A> ds_frag_p0(num_clients), ds_frag_p1(num_clients), ds_frag_p2(num_clients);

    for (uint32_t c = 0; c < num_clients; c++)
    {
        for (uint32_t k = 0; k < elements_per_client; k++)
        {
            DATATYPE v0 = 0, v1 = 0, v2 = 0;
#if PARTY == 0
            uint64_t raw = own_weight_frag[c][k];
            orthogonalize_arithmetic(&raw, &v0, 1);
#elif PARTY == 1
            uint64_t raw = own_weight_frag[c][k];
            orthogonalize_arithmetic(&raw, &v1, 1);
#elif PARTY == 2
            uint64_t raw = own_weight_frag[c][k];
            orthogonalize_arithmetic(&raw, &v2, 1);
#endif
            weight_frag_p0[c][k].template prepare_receive_from<P_0>(v0);
            weight_frag_p1[c][k].template prepare_receive_from<P_1>(v1);
            weight_frag_p2[c][k].template prepare_receive_from<P_2>(v2);
        }

        DATATYPE d0 = 0, d1 = 0, d2 = 0;
#if PARTY == 0
        uint64_t draw = own_dataset_size_frag[c];
        orthogonalize_arithmetic(&draw, &d0, 1);
#elif PARTY == 1
        uint64_t draw = own_dataset_size_frag[c];
        orthogonalize_arithmetic(&draw, &d1, 1);
#elif PARTY == 2
        uint64_t draw = own_dataset_size_frag[c];
        orthogonalize_arithmetic(&draw, &d2, 1);
#endif
        ds_frag_p0[c].template prepare_receive_from<P_0>(d0);
        ds_frag_p1[c].template prepare_receive_from<P_1>(d1);
        ds_frag_p2[c].template prepare_receive_from<P_2>(d2);
    }
    Share::communicate();
    for (uint32_t c = 0; c < num_clients; c++)
    {
        for (uint32_t k = 0; k < elements_per_client; k++)
        {
            weight_frag_p0[c][k].template complete_receive_from<P_0>();
            weight_frag_p1[c][k].template complete_receive_from<P_1>();
            weight_frag_p2[c][k].template complete_receive_from<P_2>();
        }
        ds_frag_p0[c].template complete_receive_from<P_0>();
        ds_frag_p1[c].template complete_receive_from<P_1>();
        ds_frag_p2[c].template complete_receive_from<P_2>();
    }

    // Combine the 3 native-input fragments into one properly-formed share
    // per client per element (plain local addition, no communication).
    std::vector<std::vector<A>> weight_full(num_clients, std::vector<A>(elements_per_client));
    std::vector<A> ds_full(num_clients);
    for (uint32_t c = 0; c < num_clients; c++)
    {
        for (uint32_t k = 0; k < elements_per_client; k++)
            weight_full[c][k] = weight_frag_p0[c][k] + weight_frag_p1[c][k] + weight_frag_p2[c][k];
        ds_full[c] = ds_frag_p0[c] + ds_frag_p1[c] + ds_frag_p2[c];
    }

    // Round 2: genuine secret x secret multiplication, per client per
    // element -- weight_i[k] * dataset_size_i (scalar dataset_size_i reused
    // across every element k of that same client).
    std::vector<std::vector<A>> product(num_clients, std::vector<A>(elements_per_client));
    for (uint32_t c = 0; c < num_clients; c++)
        for (uint32_t k = 0; k < elements_per_client; k++)
            product[c][k] = weight_full[c][k].prepare_mult(ds_full[c]);
    Share::communicate();
    for (uint32_t c = 0; c < num_clients; c++)
        for (uint32_t k = 0; k < elements_per_client; k++)
            product[c][k].complete_mult_without_trunc();  // see module docstring: complete_mult_with_trunc() is an empty stub for Trio

    // Sum products and dataset sizes across clients -- plain local addition
    // (no communication), same operator+ used above.
    std::vector<A> final_product(elements_per_client);
    for (uint32_t k = 0; k < elements_per_client; k++)
    {
        final_product[k] = product[0][k];
        for (uint32_t c = 1; c < num_clients; c++)
            final_product[k] = final_product[k] + product[c][k];
    }
    A total_dataset_size = ds_full[0];
    for (uint32_t c = 1; c < num_clients; c++)
        total_dataset_size = total_dataset_size + ds_full[c];

    // No reveal -- export this party's own raw Trio share of the final
    // summed product and summed dataset size directly. See module
    // docstring: reconstruction now happens at flo_server, never among the
    // compute parties. This needs no communicate() call at all (unlike the
    // reveal round it replaces), since each party already locally holds
    // its own share fields after Round 2's local per-client sum.
    if constexpr (std::is_same_v<Share, TRIO_LIVE_SHARE>)
    {
        std::ofstream out(output_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&elements_per_client), sizeof(elements_per_client));
        for (uint32_t k = 0; k < elements_per_client; k++)
        {
            uint64_t p1_out = static_cast<uint64_t>(final_product[k].get_share().raw_p1());
            uint64_t p2_out = static_cast<uint64_t>(final_product[k].get_share().raw_p2());
            out.write(reinterpret_cast<const char*>(&p1_out), sizeof(p1_out));
            out.write(reinterpret_cast<const char*>(&p2_out), sizeof(p2_out));
        }
        uint64_t ds_p1_out = static_cast<uint64_t>(total_dataset_size.get_share().raw_p1());
        uint64_t ds_p2_out = static_cast<uint64_t>(total_dataset_size.get_share().raw_p2());
        out.write(reinterpret_cast<const char*>(&ds_p1_out), sizeof(ds_p1_out));
        out.write(reinterpret_cast<const char*>(&ds_p2_out), sizeof(ds_p2_out));
    }

    *res = 0;
#endif
}
