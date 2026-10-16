use mccl::{Gpu, U32Array};
use std::{
    error::Error,
    io::{self, Write},
    time::Instant,
};

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

fn main() {
    if let Err(error) = run() {
        eprintln!("mccl: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("check") => {
            if args.next().is_some() {
                return Err("check takes no options".into());
            }
            check()
        }
        Some("bench") => bench(BenchOptions::parse(args)?),
        Some("difference-bench") => difference_bench(BenchOptions::parse(args)?),
        Some("bloom-check") => {
            if args.next().is_some() {
                return Err("bloom-check takes no options".into());
            }
            check_bloom(&Gpu::new()?)
        }
        Some("bloom-bench") => bloom_bench(BloomBenchOptions::parse(args)?),
        Some("help" | "--help" | "-h") | None => {
            println!("Usage:\n  mccl check\n  mccl bench [--size N | --sizes N,N,...] [--reps N] [--warmup N]\n             [--case all|disjoint|reverse-disjoint|interleaved|duplicates]\n\nSizes are total keys (A+B); suffixes K/M mean powers of 1024.\nDefaults: sizes=1024,65536,1048576,16777216, reps=30, warmup=5.\nBenchmark CSV goes to stdout; timing is CPU wall time through completed eval+sync,\nnot GPU timestamps. Inputs are resident; every sample creates a fresh lazy output.\nNative libraries must be built separately with CMake before Cargo.");
            println!("\n  mccl difference-bench [--size N | --sizes N,N,...] [--reps N] [--warmup N]\n                        [--case all|disjoint|reverse-disjoint|interleaved|duplicates]\nDifference sizes also mean A+B. Reports survivor count and completed-call wall\np50/p10/p90, including storage and count evaluation; no merge byte/rate model.");
            println!("\n  mccl bloom-check\n  mccl bloom-bench [--keys N] [--queries N] [--blocks N] [--seed N]\n                   [--reps N] [--warmup N]\nBloom defaults: keys=65536, queries=65536, blocks=8192, seed=0, reps=30,\nwarmup=5. K/M suffixes apply to counts; seed accepts decimal or 0x hexadecimal.\nReports isolated completed clear/insert/query CPU-wall latencies in microseconds.\nUploads and readbacks excluded; observed false-positive rate is not a guarantee.");
            Ok(())
        }
        Some(other) => Err(format!("unknown command {other:?}; use --help").into()),
    }
}

fn stable_merge(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] <= b[j] {
            result.push(a[i]);
            i += 1;
        } else {
            result.push(b[j]);
            j += 1;
        }
    }
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

fn stable_difference_indices(a: &[u32], b: &[u32]) -> Vec<usize> {
    let mut survivors = Vec::with_capacity(a.len());
    let mut j = 0;
    for (i, &key) in a.iter().enumerate() {
        while j < b.len() && b[j] < key {
            j += 1;
        }
        if j < b.len() && b[j] == key {
            // Consume earliest A occurrences first, retaining original indices.
            j += 1;
        } else {
            survivors.push(i);
        }
    }
    survivors
}

fn stable_difference(a: &[u32], b: &[u32]) -> Vec<u32> {
    stable_difference_indices(a, b)
        .into_iter()
        .map(|i| a[i])
        .collect()
}

fn exact(label: &str, actual: &[u32], expected: &[u32]) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label}: length {}, expected {}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    if let Some((i, (got, want))) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (x, y))| x != y)
    {
        return Err(format!("{label}: index {i}: got {got}, expected {want}").into());
    }
    Ok(())
}

fn check_case(gpu: &Gpu, name: &str, a: &[u32], b: &[u32]) -> Result<()> {
    let ga = gpu.upload(a)?;
    let gb = gpu.upload(b)?;
    let expected = stable_merge(a, b);
    exact(name, &gpu.merge(&ga, &gb)?.to_vec()?, &expected)?;
    exact(&format!("{name}/copy-a"), &gpu.copy(&ga)?.to_vec()?, a)?;
    exact(&format!("{name}/copy-b"), &gpu.copy(&gb)?.to_vec()?, b)?;
    Ok(())
}

struct Random(u64);
impl Random {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
    fn sorted(&mut self, len: usize, duplicate_heavy: bool) -> Vec<u32> {
        let mut values: Vec<u32> = (0..len)
            .map(|_| {
                let value = self.next();
                if duplicate_heavy {
                    value % 17
                } else {
                    value
                }
            })
            .collect();
        values.sort_unstable();
        values
    }
}

fn check() -> Result<()> {
    let gpu = Gpu::new()?;
    let mut cases = 0;
    let fixed: &[(&str, &[u32], &[u32])] = &[
        ("both-empty", &[], &[]),
        ("empty-a", &[], &[0, 1, u32::MAX]),
        ("empty-b", &[0, 1, u32::MAX], &[]),
        ("single-equal", &[7], &[7]),
        (
            "extreme-keys",
            &[0, 0, u32::MAX],
            &[0, 1, u32::MAX, u32::MAX],
        ),
        ("unequal", &[3], &[0, 1, 2, 3, 4, 5, 6]),
    ];
    for &(name, a, b) in fixed {
        check_case(&gpu, name, a, b)?;
        cases += 1;
    }
    // Exercise output tile tails AND source boundaries independently. A key-only
    // API checks exact values but cannot observe the provenance of equal keys.
    for total in [
        2, 3, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 8193, 65537,
    ] {
        for a_len in [0, 1, total / 3, total / 2, total - 1, total] {
            let b_len = total - a_len;
            for pattern in Pattern::ALL {
                let (a, b) = pattern.inputs(a_len, b_len);
                check_case(&gpu, &format!("{pattern}/{a_len}+{b_len}"), &a, &b)?;
                cases += 1;
            }
            let a = vec![u32::MAX; a_len];
            let b = vec![u32::MAX; b_len];
            check_case(&gpu, &format!("all-equal/{a_len}+{b_len}"), &a, &b)?;
            cases += 1;
        }
    }
    let mut random = Random(0xd1b5_4a32_d192_ed03);
    for (a_len, b_len) in [
        (1023, 1025),
        (1024, 1024),
        (1025, 1023),
        (1, 32769),
        (32769, 1),
        (10007, 737),
    ] {
        for duplicate_heavy in [false, true] {
            let a = random.sorted(a_len, duplicate_heavy);
            let b = random.sorted(b_len, duplicate_heavy);
            check_case(
                &gpu,
                &format!("random/{a_len}+{b_len}/{duplicate_heavy}"),
                &a,
                &b,
            )?;
            cases += 1;
        }
    }
    for i in 0..48 {
        let a_len = random.next() as usize % 12289;
        let b_len = random.next() as usize % 12289;
        let a = random.sorted(a_len, i % 2 == 0);
        let b = random.sorted(b_len, i % 2 == 0);
        check_case(&gpu, &format!("random-{i}/{a_len}+{b_len}"), &a, &b)?;
        cases += 1;
    }
    // Lazy graphs own their inputs, and arrays retain the GPU context.
    let retained = {
        let context = Gpu::new()?;
        let a = context.upload(&[1, 3, 5])?;
        let b = context.upload(&[2, 4, 6])?;
        context.merge(&a, &b)?
    };
    exact("retained graph", &retained.to_vec()?, &[1, 2, 3, 4, 5, 6])?;
    let other = Gpu::new()?;
    let a = gpu.upload(&[1])?;
    let b = other.upload(&[2])?;
    if gpu.merge(&a, &b).is_ok() || gpu.copy(&b).is_ok() {
        return Err("cross-context operation was accepted".into());
    }
    println!(
        "PASS: {cases} merge cases and their forced copies; retained graphs and context rejection"
    );
    check_difference(&gpu)?;
    check_bloom(&gpu)?;
    Ok(())
}

fn check_difference_case(gpu: &Gpu, name: &str, a: &[u32], b: &[u32]) -> Result<()> {
    let ga = gpu.upload(a)?;
    let gb = gpu.upload(b)?;
    let output = gpu.difference(&ga, &gb)?;
    if output.capacity() != a.len() {
        return Err(format!("{name}: difference capacity is not A length").into());
    }
    let expected = stable_difference(a, b);
    output.eval()?;
    if output.count()? != expected.len() {
        return Err(format!("{name}: difference count mismatch").into());
    }
    exact(name, &output.to_vec()?, &expected)?;
    // Repeated reads must retain both outputs and never reveal the capacity tail.
    exact(&format!("{name}/repeat"), &output.to_vec()?, &expected)
}

fn check_difference(gpu: &Gpu) -> Result<()> {
    let fixed: &[(&str, &[u32], &[u32])] = &[
        ("both-empty", &[], &[]),
        ("empty-a", &[], &[0, 1, u32::MAX]),
        ("empty-b", &[0, 1, u32::MAX], &[]),
        ("disjoint", &[0, 2, 4], &[1, 3, 5]),
        ("reverse-disjoint", &[7, 8], &[0, 1, 2]),
        ("all-cancel", &[0, 1, 1, u32::MAX], &[0, 1, 1, 1, u32::MAX]),
        (
            "duplicate-counts",
            &[0, 0, 0, 1, 1, 2, 2, 2, u32::MAX],
            &[0, 1, 1, 1, 2, u32::MAX, u32::MAX],
        ),
        ("unequal-a", &[3], &[0, 1, 2, 3, 4, 5, 6]),
        ("unequal-b", &[0, 1, 2, 3, 4, 5, 6], &[3]),
    ];
    // The u32 API cannot observe equal-key provenance, so assert the oracle's
    // earliest-cancellation rule with indices as well as checking GPU values.
    if stable_difference_indices(&[1, 1, 1, 2, 2], &[1, 2]) != [1, 2, 4] {
        return Err("difference oracle did not retain the expected indices".into());
    }
    let mut cases = 0;
    for &(name, a, b) in fixed {
        check_difference_case(gpu, name, a, b)?;
        cases += 1;
    }
    for a_len in [1023, 1024, 1025, 2049, 4097] {
        for b_len in [0, 1, 1023, 1024, 1025, 4098] {
            let a = vec![u32::MAX; a_len];
            let b = vec![u32::MAX; b_len];
            check_difference_case(gpu, &format!("equal-run/{a_len}-{b_len}"), &a, &b)?;
            cases += 1;
        }
    }
    // Multiple long runs with leading/trailing survivors exercise stable compaction.
    let mut a = vec![0; 1025];
    a.extend(vec![7; 3073]);
    a.extend(vec![u32::MAX; 1025]);
    let mut b = vec![0; 1024];
    b.extend(vec![7; 2049]);
    b.extend(vec![u32::MAX; 1024]);
    check_difference_case(gpu, "mixed-long-runs", &a, &b)?;
    cases += 1;
    let mut random = Random(0x813c_a5e9_042f_7d61);
    for (a_len, b_len) in [
        (1, 32769),
        (32769, 1),
        (1023, 1025),
        (1024, 1024),
        (1025, 1023),
        (10007, 737),
    ] {
        for duplicates in [false, true] {
            let mut a = random.sorted(a_len, duplicates);
            let mut b = random.sorted(b_len, duplicates);
            // Include extreme values even for duplicate-heavy random inputs.
            a[0] = 0;
            b[0] = 0;
            a[a_len - 1] = u32::MAX;
            b[b_len - 1] = u32::MAX;
            check_difference_case(gpu, &format!("random/{a_len}-{b_len}/{duplicates}"), &a, &b)?;
            cases += 1;
        }
    }
    for i in 0..48 {
        let a_len = random.next() as usize % 12289;
        let b_len = random.next() as usize % 12289;
        let a = random.sorted(a_len, i % 2 == 0);
        let b = random.sorted(b_len, i % 2 == 0);
        check_difference_case(gpu, &format!("random-difference-{i}"), &a, &b)?;
        cases += 1;
    }
    let retained = {
        let context = Gpu::new()?;
        let a = context.upload(&[0, 1, 1, 1, 7, u32::MAX])?;
        let b = context.upload(&[1, 1, 7])?;
        context.difference(&a, &b)?
    };
    exact(
        "retained difference graph",
        &retained.to_vec()?,
        &[0, 1, u32::MAX],
    )?;
    if retained.capacity() != 6 || retained.count()? != 3 {
        return Err("retained difference metadata mismatch".into());
    }
    let other = Gpu::new()?;
    let a = gpu.upload(&[1])?;
    let b = other.upload(&[1])?;
    if gpu.difference(&a, &b).is_ok() || gpu.difference(&b, &a).is_ok() {
        return Err("cross-context difference was accepted".into());
    }
    println!(
        "PASS: {cases} difference cases; count/capacity, retained graphs and context rejection"
    );
    Ok(())
}

#[derive(Clone, Copy)]
enum Pattern {
    Disjoint,
    ReverseDisjoint,
    Interleaved,
    Duplicates,
}
impl Pattern {
    const ALL: [Self; 4] = [
        Self::Disjoint,
        Self::ReverseDisjoint,
        Self::Interleaved,
        Self::Duplicates,
    ];

    fn parse(value: &str) -> Result<Self> {
        match value {
            "disjoint" => Ok(Self::Disjoint),
            "reverse-disjoint" => Ok(Self::ReverseDisjoint),
            "interleaved" => Ok(Self::Interleaved),
            "duplicates" => Ok(Self::Duplicates),
            _ => Err(format!("unknown case {value:?}").into()),
        }
    }

    fn inputs(self, a_len: usize, b_len: usize) -> (Vec<u32>, Vec<u32>) {
        match self {
            Self::Disjoint => (
                (0..a_len).map(|i| i as u32).collect(),
                (0..b_len).map(|i| (a_len + i) as u32).collect(),
            ),
            Self::ReverseDisjoint => (
                (0..a_len).map(|i| (b_len + i) as u32).collect(),
                (0..b_len).map(|i| i as u32).collect(),
            ),
            Self::Interleaved => (
                (0..a_len).map(|i| (2 * i) as u32).collect(),
                (0..b_len).map(|i| (2 * i + 1) as u32).collect(),
            ),
            Self::Duplicates => (
                (0..a_len).map(|i| (i / 32) as u32).collect(),
                (0..b_len).map(|i| (i / 32) as u32).collect(),
            ),
        }
    }
}
impl std::fmt::Display for Pattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Disjoint => "disjoint",
            Self::ReverseDisjoint => "reverse-disjoint",
            Self::Interleaved => "interleaved",
            Self::Duplicates => "duplicates",
        })
    }
}

struct BenchOptions {
    sizes: Vec<usize>,
    reps: usize,
    warmup: usize,
    patterns: Vec<Pattern>,
}
impl BenchOptions {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self> {
        let mut result = Self {
            sizes: vec![1024, 65536, 1048576, 16777216],
            reps: 30,
            warmup: 5,
            patterns: Pattern::ALL.to_vec(),
        };
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--size" => result.sizes = vec![parse_size(&value)?],
                "--sizes" => {
                    result.sizes = value.split(',').map(parse_size).collect::<Result<_>>()?
                }
                "--reps" => result.reps = value.parse()?,
                "--warmup" => result.warmup = value.parse()?,
                "--case" => {
                    result.patterns = if value == "all" {
                        Pattern::ALL.to_vec()
                    } else {
                        vec![Pattern::parse(&value)?]
                    }
                }
                _ => return Err(format!("unknown benchmark option {flag:?}").into()),
            }
        }
        if result.reps == 0 || result.warmup == 0 {
            return Err("reps and warmup must both be nonzero".into());
        }
        Ok(result)
    }
}

fn parse_size(value: &str) -> Result<usize> {
    let (digits, scale) = if let Some(n) = value.strip_suffix(['m', 'M']) {
        (n, 1024 * 1024)
    } else if let Some(n) = value.strip_suffix(['k', 'K']) {
        (n, 1024)
    } else {
        (value, 1)
    };
    let size = digits
        .parse::<usize>()?
        .checked_mul(scale)
        .ok_or("size overflow")?;
    if size == 0 || size > i32::MAX as usize {
        return Err("benchmark sizes must be in 1..=INT_MAX".into());
    }
    Ok(size)
}

fn sample(create: impl FnOnce() -> mccl::Result<U32Array>) -> Result<f64> {
    let start = Instant::now();
    let output = create()?;
    output.eval()?;
    let seconds = start.elapsed().as_secs_f64();
    // Node destruction is outside the interval. Returning its allocation to MLX's
    // pool allows the next fresh node to reuse it, identically for merge and copy.
    drop(output);
    Ok(seconds)
}

struct Stats {
    median: f64,
    p10: f64,
    p90: f64,
}
impl Stats {
    fn new(mut values: Vec<f64>) -> Self {
        values.sort_unstable_by(f64::total_cmp);
        let quantile = |p: f64| {
            let position = p * (values.len() - 1) as f64;
            let lo = position.floor() as usize;
            let hi = position.ceil() as usize;
            values[lo] + (values[hi] - values[lo]) * (position - lo as f64)
        };
        Self {
            median: quantile(0.5),
            p10: quantile(0.1),
            p90: quantile(0.9),
        }
    }
}

fn bench(options: BenchOptions) -> Result<()> {
    let gpu = Gpu::new()?;
    let stdout = io::stdout();
    let mut csv = io::BufWriter::new(stdout.lock());
    writeln!(csv, "case,total_keys,a_keys,b_keys,operation,reps,warmup,timing,logical_bytes,median_us,p10_us,p90_us,effective_gb_s,copy_over_merge_ratio")?;
    eprintln!("Timing CPU wall completed calls (fresh graph + eval + stream sync); uploads/readback/drop excluded.\nLogical bytes = 8*N for both operations; effective GB/s is not measured DRAM bandwidth.\nMLX allocator pool enabled/reused; copy and merge run in alternating order after warmup.");
    for total in options.sizes {
        let a_len = total / 2;
        let b_len = total - a_len;
        for &pattern in &options.patterns {
            let (a, b) = pattern.inputs(a_len, b_len);
            let expected = stable_merge(&a, &b);
            let ga = gpu.upload(&a)?;
            let gb = gpu.upload(&b)?;
            // Same total N and u32 type. Actual forced dispatch initializes every
            // input buffer before timing; the unified-memory host allocation alone
            // is not treated as sufficient proof of GPU residency/initialization.
            let copy_input = gpu.upload(&expected)?;
            drop(a);
            drop(b);
            exact(
                "benchmark merge",
                &gpu.merge(&ga, &gb)?.to_vec()?,
                &expected,
            )?;
            exact(
                "benchmark copy",
                &gpu.copy(&copy_input)?.to_vec()?,
                &expected,
            )?;
            drop(expected);
            for _ in 0..options.warmup {
                gpu.merge(&ga, &gb)?.eval()?;
                gpu.copy(&copy_input)?.eval()?;
            }
            let mut merges = Vec::with_capacity(options.reps);
            let mut copies = Vec::with_capacity(options.reps);
            for repetition in 0..options.reps {
                if repetition % 2 == 0 {
                    merges.push(sample(|| gpu.merge(&ga, &gb))?);
                    copies.push(sample(|| gpu.copy(&copy_input))?);
                } else {
                    copies.push(sample(|| gpu.copy(&copy_input))?);
                    merges.push(sample(|| gpu.merge(&ga, &gb))?);
                }
            }
            let merge = Stats::new(merges);
            let copy = Stats::new(copies);
            let ratio = copy.median / merge.median;
            for (name, stats) in [("merge", &merge), ("copy", &copy)] {
                writeln!(csv, "{pattern},{total},{a_len},{b_len},{name},{},{},cpu_wall_completed_call,{},{:.3},{:.3},{:.3},{:.3},{ratio:.6}",
                    options.reps, options.warmup, total as u64 * 8,
                    stats.median * 1e6, stats.p10 * 1e6, stats.p90 * 1e6,
                    total as f64 * 8.0 / stats.median / 1e9)?;
            }
            csv.flush()?;
        }
    }
    Ok(())
}

fn difference_bench(options: BenchOptions) -> Result<()> {
    let gpu = Gpu::new()?;
    let stdout = io::stdout();
    let mut csv = io::BufWriter::new(stdout.lock());
    writeln!(
        csv,
        "case,N_A,N_B,count,reps,warmup,timing,p50_us,p10_us,p90_us"
    )?;
    eprintln!("Difference timing: CPU wall time for a fresh lazy graph + full storage/count eval + stream sync.\nInputs device-resident and warmed; upload, count readback, exact value readback and destruction excluded.\nNo logical-byte bandwidth or merge/copy ratio is assigned to this multi-stage operation.");
    for total in options.sizes {
        let a_len = total / 2;
        let b_len = total - a_len;
        for &pattern in &options.patterns {
            let (a, b) = pattern.inputs(a_len, b_len);
            let expected = stable_difference(&a, &b);
            let ga = gpu.upload(&a)?;
            let gb = gpu.upload(&b)?;
            drop(a);
            drop(b);
            for _ in 0..options.warmup {
                gpu.difference(&ga, &gb)?.eval()?;
            }
            let mut samples = Vec::with_capacity(options.reps);
            for _ in 0..options.reps {
                let start = Instant::now();
                let output = gpu.difference(&ga, &gb)?;
                output.eval()?;
                samples.push(start.elapsed().as_secs_f64());
                // Check the actual timed output, but keep all host reads outside
                // the completed-call interval and never read its capacity tail.
                if output.count()? != expected.len() {
                    return Err("benchmark difference count mismatch".into());
                }
                exact("benchmark difference", &output.to_vec()?, &expected)?;
                drop(output);
            }
            let stats = Stats::new(samples);
            writeln!(
                csv,
                "{pattern},{a_len},{b_len},{},{},{},cpu_wall_completed_call,{:.3},{:.3},{:.3}",
                expected.len(),
                options.reps,
                options.warmup,
                stats.median * 1e6,
                stats.p10 * 1e6,
                stats.p90 * 1e6
            )?;
            csv.flush()?;
        }
    }
    Ok(())
}

// Byte-oriented xxHash64 short-input reference, deliberately distinct from the
// Metal u32-specialized path. All Bloom keys enter as four little-endian bytes.
fn reference_xxhash64(bytes: &[u8], seed: u64) -> u64 {
    const P1: u64 = 11_400_714_785_074_694_791;
    const P2: u64 = 14_029_467_366_897_019_727;
    const P3: u64 = 1_609_587_929_392_839_161;
    const P4: u64 = 9_650_029_242_287_828_579;
    const P5: u64 = 2_870_177_450_012_600_261;
    assert!(bytes.len() < 32);
    let mut hash = seed.wrapping_add(P5).wrapping_add(bytes.len() as u64);
    let (chunks, mut remaining) = bytes.as_chunks::<8>();
    for chunk in chunks {
        let lane = u64::from_le_bytes(*chunk)
            .wrapping_mul(P2)
            .rotate_left(31)
            .wrapping_mul(P1);
        hash = (hash ^ lane)
            .rotate_left(27)
            .wrapping_mul(P1)
            .wrapping_add(P4);
    }
    if remaining.len() >= 4 {
        let lane = u32::from_le_bytes(remaining[..4].try_into().unwrap()) as u64;
        hash = (hash ^ lane.wrapping_mul(P1))
            .rotate_left(23)
            .wrapping_mul(P2)
            .wrapping_add(P3);
        remaining = &remaining[4..];
    }
    for &byte in remaining {
        hash = (hash ^ (byte as u64).wrapping_mul(P5))
            .rotate_left(11)
            .wrapping_mul(P1);
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(P2);
    hash ^= hash >> 29;
    hash = hash.wrapping_mul(P3);
    hash ^ (hash >> 32)
}

fn bloom_locations(key: u32, seed: u64, num_blocks: u32) -> (usize, [u32; 8]) {
    let hash = reference_xxhash64(&key.to_le_bytes(), seed);
    let block = (((hash >> 32) * num_blocks as u64) >> 32) as usize;
    let salts: [u32; 8] = [
        0x47b6137b, 0x44974d91, 0x8824ad5b, 0xa2b7289d, 0x705495c7, 0x2df1424b, 0x9efc4947,
        0x05c6fb31,
    ];
    (
        block * 8,
        salts.map(|salt| 1_u32 << salt.wrapping_mul(hash as u32).wrapping_shr(27)),
    )
}

fn reference_bloom_add(words: &mut [u32], keys: &[u32], seed: u64) {
    let blocks = (words.len() / 8) as u32;
    for &key in keys {
        let (offset, masks) = bloom_locations(key, seed, blocks);
        for (word, mask) in words[offset..offset + 8].iter_mut().zip(masks) {
            *word |= mask;
        }
    }
}

fn reference_bloom_query(words: &[u32], keys: &[u32], seed: u64) -> Vec<u32> {
    keys.iter()
        .map(|&key| {
            let (offset, masks) = bloom_locations(key, seed, (words.len() / 8) as u32);
            u32::from(
                words[offset..offset + 8]
                    .iter()
                    .zip(masks)
                    .all(|(&word, mask)| word & mask != 0),
            )
        })
        .collect()
}

fn check_bloom_oracle() -> Result<()> {
    // Independent fixtures generated with Python xxhash==3.6.0, xxh64 on
    // key.to_bytes(4, "little"); not derived from the GPU implementation.
    for (key, seed, expected) in [
        (0_u32, 0_u64, 4_246_796_580_750_024_372_u64),
        (1, 0, 17_595_444_997_414_146_897),
        (2, 0, 2_844_853_711_750_869_662),
        (0, 1, 3_292_434_176_395_979_456),
        (u32::MAX, 1, 4_365_729_893_815_087_465),
        (0, u64::MAX, 9_365_271_485_936_865_770),
        (u32::MAX, u64::MAX, 643_262_203_274_103_293),
    ] {
        if reference_xxhash64(&key.to_le_bytes(), seed) != expected {
            return Err(format!("xxHash64 oracle fixture failed for {key}, seed={seed}").into());
        }
    }
    Ok(())
}

fn check_bloom_case(gpu: &Gpu, blocks: u32, seed: u64, keys: &[u32]) -> Result<()> {
    let label = format!("bloom/blocks={blocks}/seed={seed}/keys={}", keys.len());
    let mut filter = gpu.bloom(blocks, seed)?;
    if filter.num_blocks() != blocks || filter.seed() != seed {
        return Err(format!("{label}: metadata mismatch").into());
    }
    let empty = gpu.upload(&[])?;
    let input = gpu.upload(keys)?;
    let mut expected_words = vec![0; blocks as usize * 8];
    exact(&label, &filter.words()?, &expected_words)?;
    filter.add(&empty)?;
    exact(&label, &filter.contains(&empty)?.to_vec()?, &[])?;
    // Do not read this query until after mutation: lazy aliasing would change it.
    let before_add = filter.contains(&input)?;
    let midpoint = keys.len() / 2;
    for chunk in [&keys[..midpoint], &keys[midpoint..]] {
        let uploaded = gpu.upload(chunk)?;
        // Exercise a lazy source rather than only already-completed uploads.
        filter.add(&gpu.copy(&uploaded)?)?;
        reference_bloom_add(&mut expected_words, chunk, seed);
        exact(&label, &filter.words()?, &expected_words)?;
        exact(
            &label,
            &filter.contains(&uploaded)?.to_vec()?,
            &vec![1; chunk.len()],
        )?;
    }
    exact(&label, &before_add.to_vec()?, &vec![0; keys.len()])?;
    filter.add(&input)?;
    exact(
        &format!("{label}/duplicate"),
        &filter.words()?,
        &expected_words,
    )?;
    let all_present = filter.contains(&gpu.copy(&input)?)?;
    let probes: Vec<u32> = (0..2049_u32)
        .map(|key| key.wrapping_mul(2_654_435_761))
        .collect();
    let uploaded_probes = gpu.upload(&probes)?;
    exact(
        &label,
        &filter.contains(&uploaded_probes)?.to_vec()?,
        &reference_bloom_query(&expected_words, &probes, seed),
    )?;
    filter.clear()?;
    exact(
        &format!("{label}/clear"),
        &filter.words()?,
        &vec![0; expected_words.len()],
    )?;
    exact(
        &format!("{label}/snapshot-before-reinsert"),
        &all_present.to_vec()?,
        &vec![1; keys.len()],
    )?;
    let after_clear = filter.contains(&input)?;
    filter.add(&input)?;
    exact(&label, &filter.words()?, &expected_words)?;
    exact(
        &format!("{label}/old-positive"),
        &all_present.to_vec()?,
        &vec![1; keys.len()],
    )?;
    exact(
        &format!("{label}/old-negative"),
        &after_clear.to_vec()?,
        &vec![0; keys.len()],
    )?;
    Ok(())
}

fn check_bloom(gpu: &Gpu) -> Result<()> {
    check_bloom_oracle()?;
    for blocks in [0, i32::MAX as u32 / 8 + 1, u32::MAX] {
        if gpu.bloom(blocks, 0).is_ok() {
            return Err("invalid Bloom block count accepted".into());
        }
    }
    let mut random = Random(0xb100_0000_cafe_1234);
    let random_keys: Vec<u32> = (0..4097).map(|_| random.next()).collect();
    let mut cases = 0;
    for seed in [0, 1, u64::MAX] {
        for blocks in [1, 3, 13, 256] {
            for keys in [
                Vec::new(),
                vec![u32::MAX],
                vec![0, 1, u32::MAX, 1 << 31, u32::MAX - 1, 0],
                vec![17; 8193],
                (0..1025).collect(),
                random_keys.clone(),
            ] {
                check_bloom_case(gpu, blocks, seed, &keys)?;
                cases += 1;
            }
            // Distinct keys all targeting one block stress concurrent atomic OR.
            let contended: Vec<u32> = (0..u32::MAX)
                .filter(|&key| bloom_locations(key, seed, blocks).0 == 0)
                .take(1025)
                .collect();
            check_bloom_case(gpu, blocks, seed, &contended)?;
            cases += 1;

            // Even inserted keys and odd probes are disjoint by construction.
            // Compare the exact empirical false-positive count, not an FPR bound.
            let inserted: Vec<u32> = (0..128).map(|n| n * 2).collect();
            let absent: Vec<u32> = (0..4096).map(|n| n * 2 + 1).collect();
            let mut filter = gpu.bloom(blocks, seed)?;
            let mut words = vec![0; blocks as usize * 8];
            reference_bloom_add(&mut words, &inserted, seed);
            filter.add(&gpu.upload(&inserted)?)?;
            exact("FPR bitset", &filter.words()?, &words)?;
            exact(
                "FPR inserted keys",
                &filter.contains(&gpu.upload(&inserted)?)?.to_vec()?,
                &vec![1; inserted.len()],
            )?;
            let oracle = reference_bloom_query(&words, &absent, seed);
            let flags = filter.contains(&gpu.upload(&absent)?)?.to_vec()?;
            exact("deterministic false positives", &flags, &oracle)?;
            let false_positives = flags.iter().filter(|&&flag| flag == 1).count();
            let oracle_count = oracle.iter().filter(|&&flag| flag == 1).count();
            if false_positives != oracle_count {
                return Err("empirical Bloom false-positive count mismatch".into());
            }
        }
    }
    let input = gpu.upload(&[0, u32::MAX])?;
    let mut first = gpu.bloom(13, 1)?;
    let mut second = gpu.bloom(13, 1)?;
    first.add(&input)?;
    exact("independent empty filter", &second.words()?, &[0; 104])?;
    second.add(&input)?;
    let second_words = second.words()?;
    first.clear()?;
    exact(
        "independent populated filter",
        &second.words()?,
        &second_words,
    )?;
    let other = Gpu::new()?;
    let foreign = other.upload(&[1])?;
    if first.add(&foreign).is_ok() || first.contains(&foreign).is_ok() {
        return Err("Bloom cross-context operation was accepted".into());
    }
    exact("rejected insert unchanged", &first.words()?, &[0; 104])?;
    // Both the filter and its completed output outlive the original Gpu handle.
    let (mut retained, retained_input) = {
        let context = Gpu::new()?;
        (
            context.bloom(3, u64::MAX)?,
            context.upload(&[0, 19, u32::MAX])?,
        )
    };
    retained.add(&retained_input)?;
    let retained_output = retained.contains(&retained_input)?;
    retained.clear()?;
    drop(retained);
    drop(retained_input);
    exact(
        "retained Bloom query",
        &retained_output.to_vec()?,
        &[1, 1, 1],
    )?;
    println!("PASS: {cases} Bloom cases; exact CPU bits, hash fixtures, no false negatives,\n deterministic empirical false positives, mutation snapshots, independent filters and context lifetime");
    Ok(())
}

struct BloomBenchOptions {
    keys: usize,
    queries: usize,
    blocks: u32,
    seed: u64,
    reps: usize,
    warmup: usize,
}

impl BloomBenchOptions {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self> {
        let mut options = Self {
            keys: 65536,
            queries: 65536,
            blocks: 8192,
            seed: 0,
            reps: 30,
            warmup: 5,
        };
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--keys" => options.keys = parse_size(&value)?,
                "--queries" => options.queries = parse_size(&value)?,
                "--blocks" => options.blocks = u32::try_from(parse_size(&value)?)?,
                "--seed" => {
                    options.seed = if let Some(hex) = value.strip_prefix("0x") {
                        u64::from_str_radix(hex, 16)?
                    } else {
                        value.parse()?
                    }
                }
                "--reps" => options.reps = value.parse()?,
                "--warmup" => options.warmup = value.parse()?,
                _ => return Err(format!("unknown Bloom benchmark option {flag:?}").into()),
            }
        }
        if options.blocks == 0 || options.blocks > i32::MAX as u32 / 8 {
            return Err("Bloom blocks must be in 1..=INT_MAX/8".into());
        }
        if options.reps == 0 || options.warmup == 0 {
            return Err("reps and warmup must both be nonzero".into());
        }
        if options.queries < 2 {
            return Err(
                "Bloom benchmark requires at least two queries (one hit and one absent)".into(),
            );
        }
        Ok(options)
    }
}

fn bloom_bench(options: BloomBenchOptions) -> Result<()> {
    check_bloom_oracle()?;
    let gpu = Gpu::new()?;
    let mut filter = gpu.bloom(options.blocks, options.seed)?;
    // Unique evens are members. Odd queries are guaranteed nonmembers; even
    // queries cycle through the members. No host membership lookup is timed.
    let keys: Vec<u32> = (0..options.keys).map(|n| (n as u32) * 2).collect();
    let queries: Vec<u32> = (0..options.queries)
        .map(|n| {
            if n % 2 == 0 {
                ((n / 2 % options.keys) as u32) * 2
            } else {
                (n as u32) | 1
            }
        })
        .collect();
    let input = gpu.upload(&keys)?;
    let query_input = gpu.upload(&queries)?;
    let mut expected_words = vec![0; options.blocks as usize * 8];
    reference_bloom_add(&mut expected_words, &keys, options.seed);
    let expected_flags = reference_bloom_query(&expected_words, &queries, options.seed);
    for _ in 0..options.warmup {
        filter.clear()?;
        filter.add(&input)?;
        drop(filter.contains(&query_input)?);
    }
    eprintln!("Bloom timing: isolated CPU-wall completed clear/insert/query calls; inputs device-resident.\nCreation, upload, readback, oracle work, and output destruction excluded. Insert starts cleared;\nquery sees the completed insertion. No bandwidth model. Observed FPR is empirical, not a guarantee.");
    let mut clears = Vec::with_capacity(options.reps);
    let mut inserts = Vec::with_capacity(options.reps);
    let mut query_times = Vec::with_capacity(options.reps);
    let mut false_positives = 0;
    let mut true_negatives = 0;
    for _ in 0..options.reps {
        let start = Instant::now();
        filter.clear()?;
        clears.push(start.elapsed().as_secs_f64());
        let start = Instant::now();
        filter.add(&input)?;
        inserts.push(start.elapsed().as_secs_f64());
        let start = Instant::now();
        let output = filter.contains(&query_input)?;
        query_times.push(start.elapsed().as_secs_f64());
        // Check the actual timed output, after all three timing intervals.
        let flags = output.to_vec()?;
        exact("benchmark Bloom query", &flags, &expected_flags)?;
        exact("benchmark Bloom words", &filter.words()?, &expected_words)?;
        false_positives = 0;
        true_negatives = 0;
        for (i, &flag) in flags.iter().enumerate() {
            if i % 2 == 0 {
                if flag != 1 {
                    return Err("benchmark Bloom false negative".into());
                }
            } else if flag == 1 {
                false_positives += 1;
            } else {
                true_negatives += 1;
            }
        }
    }
    let absent_queries = options.queries / 2;
    let observed_fpr = false_positives as f64 / absent_queries as f64;
    let stdout = io::stdout();
    let mut csv = io::BufWriter::new(stdout.lock());
    writeln!(csv, "operation,insert_keys,query_keys,num_blocks,seed,reps,warmup,timing,p50_us,p10_us,p90_us,member_queries,absent_queries,false_positives,true_negatives,observed_fpr")?;
    for (operation, samples) in [
        ("clear", clears),
        ("insert", inserts),
        ("query", query_times),
    ] {
        let stats = Stats::new(samples);
        writeln!(csv, "{operation},{},{},{},{},{},{},cpu_wall_completed_call,{:.3},{:.3},{:.3},{},{absent_queries},{false_positives},{true_negatives},{observed_fpr:.9}",
            options.keys, options.queries, options.blocks, options.seed, options.reps, options.warmup,
            stats.median * 1e6, stats.p10 * 1e6, stats.p90 * 1e6,
            options.queries - absent_queries)?;
    }
    csv.flush()?;
    Ok(())
}
