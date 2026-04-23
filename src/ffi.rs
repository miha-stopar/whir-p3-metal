//! C FFI for benchmarking from iOS/Swift.
//!
//! Compile with `--features gpu-metal,ffi` and `crate-type = ["staticlib"]`.

use std::os::raw::c_char;
use std::time::Instant;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::Field;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::fiat_shamir::domain_separator::DomainSeparator;
use crate::gpu_dft::{GpuMmcs, MetalBabyBearDft};
use crate::parameters::{
    DEFAULT_MAX_POW, FoldingFactor, ProtocolParameters, SecurityAssumption, SumcheckStrategy,
    WhirConfig,
};
use crate::whir::committer::writer::CommitmentWriter;
use crate::whir::proof::WhirProof;
use crate::whir::prover::Prover;
use crate::GpuChallenger;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
type MyGpuMmcs = GpuMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;

#[repr(C)]
pub struct BenchResult {
    pub cpu_ms: f64,
    pub gpu_ms: f64,
    pub fused_ms: f64,
    pub grind_ms: f64,
}

fn run_cpu(n: usize, f: usize, r: usize) -> f64 {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let merkle_hash = MyHash::new(perm.clone());
    let merkle_compress = MyCompress::new(perm.clone());
    let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

    let whir_params = ProtocolParameters {
        security_level: 100,
        pow_bits: DEFAULT_MAX_POW,
        folding_factor: FoldingFactor::Constant(f),
        mmcs,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: r,
        rs_domain_initial_reduction_factor: f.min(3),
    };
    let params = WhirConfig::new(n, whir_params.clone());
    let max_fft = 1 << params.max_fft_size();
    let dft = Radix2DFTSmallBatch::<F>::new(max_fft);

    let num_coeffs = 1usize << n;
    let mut rng2 = SmallRng::seed_from_u64(0);
    let polynomial = Poly::<F>::new((0..num_coeffs).map(|_| rng2.random()).collect());
    let mut initial_statement = params.initial_statement(polynomial, SumcheckStrategy::Svo);
    let _ = initial_statement.evaluate(&Point::rand(&mut rng2, n));

    let mut domainsep = DomainSeparator::new(vec![]);
    domainsep.commit_statement::<_, _, 8>(&params);
    domainsep.add_whir_proof::<_, _, 8>(&params);

    // Warmup
    {
        let mut ch = MyChallenger::new(perm.clone());
        domainsep.observe_domain_separator(&mut ch);
        let mut proof = WhirProof::<F, EF, MyMmcs>::from_protocol_parameters(&whir_params, n);
        let comm = CommitmentWriter::new(&params);
        let mut s = initial_statement.clone();
        let pd = comm.commit(&dft, &mut proof, &mut ch, &mut s).unwrap();
        Prover(&params)
            .prove(&dft, &mut proof, &mut ch, &s, pd)
            .unwrap();
    }

    let mut times = Vec::new();
    for _ in 0..3 {
        let t0 = Instant::now();
        let mut ch = MyChallenger::new(perm.clone());
        domainsep.observe_domain_separator(&mut ch);
        let mut proof = WhirProof::<F, EF, MyMmcs>::from_protocol_parameters(&whir_params, n);
        let comm = CommitmentWriter::new(&params);
        let mut s = initial_statement.clone();
        let pd = comm.commit(&dft, &mut proof, &mut ch, &mut s).unwrap();
        Prover(&params)
            .prove(&dft, &mut proof, &mut ch, &s, pd)
            .unwrap();
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[1] * 1000.0
}

fn run_gpu_fused(n: usize, f: usize, r: usize, use_gpu_grind: bool) -> f64 {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let merkle_hash = MyHash::new(perm.clone());
    let merkle_compress = MyCompress::new(perm.clone());
    let inner = MyMmcs::new(merkle_hash, merkle_compress, 0);

    let whir_params_cpu = ProtocolParameters {
        security_level: 100,
        pow_bits: DEFAULT_MAX_POW,
        folding_factor: FoldingFactor::Constant(f),
        mmcs: inner.clone(),
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: r,
        rs_domain_initial_reduction_factor: f.min(3),
    };
    let params_cpu = WhirConfig::<EF, F, MyMmcs, MyChallenger>::new(n, whir_params_cpu);
    let max_fft = 1 << params_cpu.max_fft_size();
    let dft = MetalBabyBearDft::new(max_fft);
    let mmcs = MyGpuMmcs::new(inner, dft.clone());

    if use_gpu_grind {
        let whir_params = ProtocolParameters {
            security_level: 100,
            pow_bits: DEFAULT_MAX_POW,
            folding_factor: FoldingFactor::Constant(f),
            mmcs,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: r,
            rs_domain_initial_reduction_factor: f.min(3),
        };
        let params =
            WhirConfig::<EF, F, MyGpuMmcs, GpuChallenger>::new(n, whir_params.clone());

        let num_coeffs = 1usize << n;
        let mut rng2 = SmallRng::seed_from_u64(0);
        let polynomial = Poly::<F>::new((0..num_coeffs).map(|_| rng2.random()).collect());
        let mut initial_statement =
            params.initial_statement(polynomial, SumcheckStrategy::Svo);
        let _ = initial_statement.evaluate(&Point::rand(&mut rng2, n));

        let mut domainsep = DomainSeparator::new(vec![]);
        domainsep.commit_statement::<_, _, 8>(&params);
        domainsep.add_whir_proof::<_, _, 8>(&params);

        {
            let mut ch = GpuChallenger::new(perm.clone(), dft.clone());
            domainsep.observe_domain_separator(&mut ch);
            let mut proof =
                WhirProof::<F, EF, MyGpuMmcs>::from_protocol_parameters(&whir_params, n);
            let comm = CommitmentWriter::new(&params);
            let mut s = initial_statement.clone();
            let pd = comm.commit_fused(&dft, &mut proof, &mut ch, &mut s).unwrap();
            Prover(&params)
                .prove_fused(&dft, &mut proof, &mut ch, &s, pd)
                .unwrap();
        }

        let mut times = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let mut ch = GpuChallenger::new(perm.clone(), dft.clone());
            domainsep.observe_domain_separator(&mut ch);
            let mut proof =
                WhirProof::<F, EF, MyGpuMmcs>::from_protocol_parameters(&whir_params, n);
            let comm = CommitmentWriter::new(&params);
            let mut s = initial_statement.clone();
            let pd = comm.commit_fused(&dft, &mut proof, &mut ch, &mut s).unwrap();
            Prover(&params)
                .prove_fused(&dft, &mut proof, &mut ch, &s, pd)
                .unwrap();
            times.push(t0.elapsed().as_secs_f64());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[1] * 1000.0
    } else {
        let whir_params = ProtocolParameters {
            security_level: 100,
            pow_bits: DEFAULT_MAX_POW,
            folding_factor: FoldingFactor::Constant(f),
            mmcs,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: r,
            rs_domain_initial_reduction_factor: f.min(3),
        };
        let params = WhirConfig::new(n, whir_params.clone());

        let num_coeffs = 1usize << n;
        let mut rng2 = SmallRng::seed_from_u64(0);
        let polynomial = Poly::<F>::new((0..num_coeffs).map(|_| rng2.random()).collect());
        let mut initial_statement =
            params.initial_statement(polynomial, SumcheckStrategy::Svo);
        let _ = initial_statement.evaluate(&Point::rand(&mut rng2, n));

        let mut domainsep = DomainSeparator::new(vec![]);
        domainsep.commit_statement::<_, _, 8>(&params);
        domainsep.add_whir_proof::<_, _, 8>(&params);

        {
            let mut ch = MyChallenger::new(perm.clone());
            domainsep.observe_domain_separator(&mut ch);
            let mut proof =
                WhirProof::<F, EF, MyGpuMmcs>::from_protocol_parameters(&whir_params, n);
            let comm = CommitmentWriter::new(&params);
            let mut s = initial_statement.clone();
            let pd = comm.commit_fused(&dft, &mut proof, &mut ch, &mut s).unwrap();
            Prover(&params)
                .prove_fused(&dft, &mut proof, &mut ch, &s, pd)
                .unwrap();
        }

        let mut times = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let mut ch = MyChallenger::new(perm.clone());
            domainsep.observe_domain_separator(&mut ch);
            let mut proof =
                WhirProof::<F, EF, MyGpuMmcs>::from_protocol_parameters(&whir_params, n);
            let comm = CommitmentWriter::new(&params);
            let mut s = initial_statement.clone();
            let pd = comm.commit_fused(&dft, &mut proof, &mut ch, &mut s).unwrap();
            Prover(&params)
                .prove_fused(&dft, &mut proof, &mut ch, &s, pd)
                .unwrap();
            times.push(t0.elapsed().as_secs_f64());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[1] * 1000.0
    }
}

/// Run a single benchmark config. Returns results in milliseconds.
/// `mode`: 0 = all modes, 1 = CPU only, 2 = GPU fused only, 3 = GPU grind only.
/// Returns -1.0 for modes that fail or are skipped.
#[unsafe(no_mangle)]
pub extern "C" fn whir_bench(n: u32, fold: u32, rate: u32, mode: u32) -> BenchResult {
    let n = n as usize;
    let f = fold as usize;
    let r = rate as usize;

    let cpu_ms = if mode == 0 || mode == 1 {
        std::panic::catch_unwind(|| run_cpu(n, f, r)).unwrap_or(-1.0)
    } else {
        -1.0
    };

    let fused_ms = if mode == 0 || mode == 2 {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_gpu_fused(n, f, r, false)
        }))
        .unwrap_or(-1.0)
    } else {
        -1.0
    };

    let grind_ms = if mode == 0 || mode == 3 {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_gpu_fused(n, f, r, true)
        }))
        .unwrap_or(-1.0)
    } else {
        -1.0
    };

    BenchResult {
        cpu_ms,
        gpu_ms: -1.0,
        fused_ms,
        grind_ms,
    }
}

/// Returns a static string with device info (for display in the app).
#[unsafe(no_mangle)]
pub extern "C" fn whir_device_info() -> *const c_char {
    static INFO: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();
    let info = INFO.get_or_init(|| {
        let metal_device = metal::Device::system_default()
            .map(|d| d.name().to_string())
            .unwrap_or_else(|| "No Metal device".to_string());
        std::ffi::CString::new(format!("Metal: {metal_device}")).unwrap()
    });
    info.as_ptr()
}
