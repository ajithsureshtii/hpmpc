#pragma once

// FedAvg secure aggregation for Flotilla (see
// flotilla/docs/secure_aggregation/hpmpc_backend.md). Computes the sum of N
// clients' already-secret-shared, pre-weighted (by their own dataset size)
// updates -- but, unlike an earlier version of this program, does NOT
// reveal that sum among the compute parties. Instead each party exports
// its OWN raw share of the sum, and reconstruction happens entirely at
// flo_server (see server/secure_agg/reconstruct.py and
// aggregator_secure_mpc.py) -- so no compute party, individually or
// collectively, ever learns the plaintext aggregate. See
// docs/secure_aggregation/threat_model.md for the resulting trust-model
// consequence (flo_server becomes the one place plaintext is ever
// computed, unlike the old symmetric-reveal design). Division by the
// round's total weight also happens in plaintext at flo_server, after
// reconstruction, exactly as it did after reveal before.
//
// Since Python already pre-sums every client's share before this program
// ever runs (see "why summing happens in Python" below), and reveal is now
// gone, this program does ZERO inter-party communication for the
// client_side weighting mode -- it is a pure local read-transform-write.
// It is still built and invoked as a compiled hpmpc program, for
// consistency with mult_fedavg_secure_aggregation.hpp (which DOES still
// need real communication for its multiply step) and in case a future
// variant needs this program to do real MPC work again.
//
// Supports PROTOCOL=2 (Replicated 3PC), PROTOCOL=5 (Trio), and PROTOCOL=8
// (Tetrad) behind the SAME template -- one compiled binary per (protocol,
// party), selected entirely by which PROTOCOL/PARTY the build was invoked
// with (see hpmpc/scripts/build_secure_agg.sh).
//
// IMPORTANT, and easy to get wrong (hit during development): unlike
// Replicated_Share<DATATYPE>, which is ONE symmetric C++ class declared
// identically regardless of which PARTY compiled it (role-specific
// behavior lives inside the class via PARTY-based internal relabeling),
// Trio and Tetrad each use SEVERAL DISTINCT classes, one per real party
// role (OECL0/1/2_Share, Tetrad0/1/2/3_Share) -- and Protocols.h only
// declares the ONE class matching the PARTY this specific binary is being
// compiled as (a PARTY=1 Trio build never sees OECL0_Share or OECL2_Share
// declared at all). `if constexpr` only discards a branch's STATEMENTS; it
// does NOT suppress a name-lookup failure in the branch's own CONDITION
// EXPRESSION, so a single `if constexpr` naming all of Replicated_Share,
// OECL0_Share, OECL1_Share, and OECL2_Share together fails to even PARSE
// for any one single-party, single-protocol build (confirmed: real
// "'OECL1_Share' was not declared in this scope" compile errors when first
// building Trio). Each protocol's (and, for Trio/Tetrad, each party's)
// branch below is therefore wrapped in a `#if PROTOCOL == ...` /
// `#if PARTY == ...` preprocessor guard, so the compiler never even
// attempts to look up a role's class name in a translation unit that
// doesn't declare it -- `if constexpr` is layered on TOP of that, purely to
// handle the remaining init-vs-live Share-type distinction within an
// already protocol-and-party-narrowed branch.
//
// Unlike the tutorials (YourFirstProgram.hpp, basic_tutorial.hpp), this
// program does NOT use prepare_receive_from<P_i> to secret-share a value
// from scratch, and it does NOT use Additive_Share::operator+ to sum
// multiple clients' shares either -- see "why summing happens in Python"
// below. It reads ONE already-summed share (however many fields that
// protocol/role needs) per model-weight element from a file and reveals it.
//
// Why summing happens in Python, not here: for PROTOCOL==2 specifically,
// Replicated_Share's operator+ (Additive_Share.hpp) is specialized for
// combining a running share with a LOCALLY-DERIVED delta (e.g. the result
// of this party's own prepare_mult/prepare_dot) -- it keeps the LEFT
// operand's `x` component unchanged and SUBTRACTS the right operand's `a`
// component. That is correct for hpmpc's own internal use, but is NOT plain
// addition of two independently-secret-shared values: empirically, summing
// two independently-shared secrets this way reveals their DIFFERENCE, not
// their sum (confirmed with a minimal diagnostic during development).
// Trio's and Tetrad's own operator+ have been verified to have no such
// pitfall (plain elementwise addition on every field is correct for both --
// see hpmpc_backend.md's per-protocol sections), but shares are still
// pre-summed in Python for every protocol, for one consistent convention:
// none of these share types expose public getters, so a program here could
// not access both operands' raw fields to sum them even if it wanted to.
// Since addition of shares is local/free arithmetic in every one of these
// schemes (no communication needed, by construction), Python is just as
// correct and far simpler: backend_hpmpc.py sums every selected client's
// share fields itself (plain add-mod-2**64, elementwise) BEFORE writing the
// input file, so this program only ever has to reveal a single,
// already-combined share per element.
//
// File contract (see backend_hpmpc.py; field count/semantics per protocol
// and role documented in hpmpc_backend.md):
//   input file:  uint32 elements_per_client, then that many rows of N
//                uint64 fields each (N = 2 for Replicated and for Trio's
//                3 roles; 3 for Tetrad's 4 roles) -- the PRE-SUMMED share
//                of the weighted total, one row per flattened model-weight
//                element.
//   output file: uint32 elements_per_client, then that many rows of the
//                SAME N uint64 fields each, in the SAME order -- this
//                party's own raw share of the (still-secret) sum, in
//                exactly the on-disk shape hpmpc's own Share constructor
//                expects, so backend_hpmpc.py can pass it straight to
//                server/secure_agg/reconstruct.py without any
//                protocol-specific unpacking of its own. Output format is
//                identical to the input format precisely because no
//                computation happened here beyond what Python already did
//                (plain local addition preserves a share's field shape).
//
// This function is instantiated TWICE per protocol by protocol_executer.hpp:
// once during the init phase with a lightweight stub type used only to size
// communication buffers (Replicated_init / OECL{0,1,2}_init / etc. -- see
// PROTOCOL_INIT in Protocols.h), and once during the live phase with the
// real share type (Replicated_Share / OECL{0,1,2}_Share / etc). The
// init-phase stub types have no raw-field constructor and no raw_*()
// accessors, so the real share-construction/export logic is guarded with
// `if constexpr` per concrete live Share type -- discarded entirely, not
// just skipped, for the init-phase instantiation. Both phases run the SAME
// (zero) number of communication-bearing calls, so there is no
// init/live buffer-bookkeeping symmetry concern here anymore -- that
// concern only existed because of the reveal round this version removes.

#include "../../datatypes/Additive_Share.hpp"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <type_traits>
#include <vector>

#define RESULTTYPE DATATYPE
#define FUNCTION FedAvgSecureAggregation

// Which live Share type THIS binary's protocol+party combination uses, so
// the rest of the file only ever names ONE type per protocol family --
// never all of a role-asymmetric protocol's classes at once (see the
// module docstring above for why that fails to parse). Only meaningful
// (and only ever referenced below) for the PROTOCOL this binary was
// actually compiled with.
#if PROTOCOL == 5
    #if PARTY == 0
        #define TRIO_LIVE_SHARE OECL0_Share<DATATYPE>
    #elif PARTY == 1
        #define TRIO_LIVE_SHARE OECL1_Share<DATATYPE>
    #elif PARTY == 2
        #define TRIO_LIVE_SHARE OECL2_Share<DATATYPE>
    #endif
#endif

// Tetrad (PROTOCOL=8): 4 distinct per-role classes, same reasoning as Trio's
// above. All 4 roles' constructors take 3 raw uint64 fields in an order that
// matches this file's uniform (f0, f1, f2) on-disk layout directly --
// Tetrad0/1/2_Share(mv, l0, l1) and Tetrad3_Share(lambda1, lambda2, lambda3)
// -- see Tetrad-P_{0,1,2,3}_template.hpp and
// sharing_schemes/tetrad4pc.py's docstring for why no reordering is needed.
#if PROTOCOL == 8
    #if PARTY == 0
        #define TETRAD_LIVE_SHARE Tetrad0_Share<DATATYPE>
    #elif PARTY == 1
        #define TETRAD_LIVE_SHARE Tetrad1_Share<DATATYPE>
    #elif PARTY == 2
        #define TETRAD_LIVE_SHARE Tetrad2_Share<DATATYPE>
    #elif PARTY == 3
        #define TETRAD_LIVE_SHARE Tetrad3_Share<DATATYPE>
    #endif
#endif

template <typename Share>
void FedAvgSecureAggregation(DATATYPE* res)
{
    using A = Additive_Share<DATATYPE, Share>;

    const char* input_path = std::getenv("SECURE_AGG_INPUT_FILE");
    const char* output_path = std::getenv("SECURE_AGG_OUTPUT_FILE");
    if (input_path == nullptr || output_path == nullptr)
    {
        print("fedavg_secure_aggregation: SECURE_AGG_INPUT_FILE / SECURE_AGG_OUTPUT_FILE not set\n");
        exit(1);
    }

    uint32_t elements_per_client = 0;
    {
        std::ifstream header_in(input_path, std::ios::binary);
        if (!header_in.is_open())
        {
            print("fedavg_secure_aggregation: failed to open input file\n");
            exit(1);
        }
        header_in.read(reinterpret_cast<char*>(&elements_per_client), sizeof(elements_per_client));
    }

    std::vector<A> values(elements_per_client);

#if PROTOCOL == 2
    if constexpr (std::is_same_v<Share, Replicated_Share<DATATYPE>>)
    {
        std::ifstream in(input_path, std::ios::binary);
        uint32_t header_elements_per_client;
        in.read(reinterpret_cast<char*>(&header_elements_per_client), sizeof(header_elements_per_client));

        for (uint32_t k = 0; k < elements_per_client; k++)
        {
            uint64_t x_val = 0;
            uint64_t a_val = 0;
            in.read(reinterpret_cast<char*>(&x_val), sizeof(x_val));
            in.read(reinterpret_cast<char*>(&a_val), sizeof(a_val));
            Share raw_share(static_cast<DATATYPE>(x_val), static_cast<DATATYPE>(a_val));
            values[k] = A(raw_share);
        }

        // No reveal -- export this party's own raw (still-secret) share of
        // the sum directly. See module docstring: reconstruction now
        // happens at flo_server, never among the compute parties.
        std::ofstream out(output_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&elements_per_client), sizeof(elements_per_client));
        for (uint32_t k = 0; k < elements_per_client; k++)
        {
            uint64_t x_out = static_cast<uint64_t>(values[k].get_share().raw_x());
            uint64_t a_out = static_cast<uint64_t>(values[k].get_share().raw_a());
            out.write(reinterpret_cast<const char*>(&x_out), sizeof(x_out));
            out.write(reinterpret_cast<const char*>(&a_out), sizeof(a_out));
        }
    }
#elif PROTOCOL == 5
    // Trio. All 3 roles share the SAME 2-uint64-per-element on-disk layout
    // (p1, p2).
    if constexpr (std::is_same_v<Share, TRIO_LIVE_SHARE>)
    {
        std::ifstream in(input_path, std::ios::binary);
        uint32_t header_elements_per_client;
        in.read(reinterpret_cast<char*>(&header_elements_per_client), sizeof(header_elements_per_client));

        for (uint32_t k = 0; k < elements_per_client; k++)
        {
            uint64_t p1_val = 0;
            uint64_t p2_val = 0;
            in.read(reinterpret_cast<char*>(&p1_val), sizeof(p1_val));
            in.read(reinterpret_cast<char*>(&p2_val), sizeof(p2_val));
            Share raw_share(static_cast<DATATYPE>(p1_val), static_cast<DATATYPE>(p2_val));
            values[k] = A(raw_share);
        }

        // No reveal -- see PROTOCOL==2's branch above for why.
        std::ofstream out(output_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&elements_per_client), sizeof(elements_per_client));
        for (uint32_t k = 0; k < elements_per_client; k++)
        {
            uint64_t p1_out = static_cast<uint64_t>(values[k].get_share().raw_p1());
            uint64_t p2_out = static_cast<uint64_t>(values[k].get_share().raw_p2());
            out.write(reinterpret_cast<const char*>(&p1_out), sizeof(p1_out));
            out.write(reinterpret_cast<const char*>(&p2_out), sizeof(p2_out));
        }
    }
#elif PROTOCOL == 8
    // Tetrad. All 4 roles share the SAME 3-uint64-per-element on-disk layout
    // (f0, f1, f2) -- semantics differ by role (mv, l0, l1 for P0/1/2;
    // lambda1, lambda2, lambda3 for P3, see hpmpc_backend.md's Tetrad
    // section) but the field count/order needs no per-role branching here,
    // since each role's constructor already expects fields in this exact
    // order (see the TETRAD_LIVE_SHARE comment above). raw_mv/raw_l0/raw_l1
    // (P0/1/2) and raw_l1/raw_l2/raw_l3 (P3) are each named to match their
    // OWN role's semantics, but occupy the SAME f0/f1/f2 output slots.
    if constexpr (std::is_same_v<Share, TETRAD_LIVE_SHARE>)
    {
        std::ifstream in(input_path, std::ios::binary);
        uint32_t header_elements_per_client;
        in.read(reinterpret_cast<char*>(&header_elements_per_client), sizeof(header_elements_per_client));

        for (uint32_t k = 0; k < elements_per_client; k++)
        {
            uint64_t f0_val = 0;
            uint64_t f1_val = 0;
            uint64_t f2_val = 0;
            in.read(reinterpret_cast<char*>(&f0_val), sizeof(f0_val));
            in.read(reinterpret_cast<char*>(&f1_val), sizeof(f1_val));
            in.read(reinterpret_cast<char*>(&f2_val), sizeof(f2_val));
            Share raw_share(static_cast<DATATYPE>(f0_val), static_cast<DATATYPE>(f1_val), static_cast<DATATYPE>(f2_val));
            values[k] = A(raw_share);
        }

        // No reveal -- see PROTOCOL==2's branch above for why.
        std::ofstream out(output_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&elements_per_client), sizeof(elements_per_client));
        for (uint32_t k = 0; k < elements_per_client; k++)
        {
#if PARTY == 3
            uint64_t f0_out = static_cast<uint64_t>(values[k].get_share().raw_l1());
            uint64_t f1_out = static_cast<uint64_t>(values[k].get_share().raw_l2());
            uint64_t f2_out = static_cast<uint64_t>(values[k].get_share().raw_l3());
#else
            uint64_t f0_out = static_cast<uint64_t>(values[k].get_share().raw_mv());
            uint64_t f1_out = static_cast<uint64_t>(values[k].get_share().raw_l0());
            uint64_t f2_out = static_cast<uint64_t>(values[k].get_share().raw_l1());
#endif
            out.write(reinterpret_cast<const char*>(&f0_out), sizeof(f0_out));
            out.write(reinterpret_cast<const char*>(&f1_out), sizeof(f1_out));
            out.write(reinterpret_cast<const char*>(&f2_out), sizeof(f2_out));
        }
    }
#endif

    *res = 0;
}
