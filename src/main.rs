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
        Some("hashmap-check") => {
            if args.next().is_some() {
                return Err("hashmap-check takes no options".into());
            }
            check_hashmap(&Gpu::new()?)
        }
        Some("hashmap-bench") => hashmap_bench(MapBenchOptions::parse(args)?),
        Some("help" | "--help" | "-h") | None => {
            println!("Usage:\n  mccl check\n  mccl bench [--size N | --sizes N,N,...] [--reps N] [--warmup N]\n             [--case all|disjoint|reverse-disjoint|interleaved|duplicates]\n\nSizes are total keys (A+B); suffixes K/M mean powers of 1024.\nDefaults: sizes=1024,65536,1048576,16777216, reps=30, warmup=5.\nBenchmark CSV goes to stdout; timing is CPU wall time through completed eval+sync,\nnot GPU timestamps. Inputs are resident; every sample creates a fresh lazy output.\nNative libraries must be built separately with CMake before Cargo.");
            println!("\n  mccl difference-bench [--size N | --sizes N,N,...] [--reps N] [--warmup N]\n                        [--case all|disjoint|reverse-disjoint|interleaved|duplicates]\nDifference sizes also mean A+B. Reports survivor count and completed-call wall\np50/p10/p90, including storage and count evaluation; no merge byte/rate model.");
            println!("\n  mccl bloom-check\n  mccl bloom-bench [--keys N] [--queries N] [--blocks N] [--seed N]\n                   [--reps N] [--warmup N]\nBloom defaults: keys=65536, queries=65536, blocks=8192, seed=0, reps=30,\nwarmup=5. K/M suffixes apply to counts; seed accepts decimal or 0x hexadecimal.\nReports isolated completed clear/insert/query CPU-wall latencies in microseconds.\nUploads and readbacks excluded; observed false-positive rate is not a guarantee.");
            println!("\n  mccl hashmap-check\n  mccl hashmap-bench [--keys N] [--batch N] [--queries N] [--capacity N]\n                     [--scenario cold-insert|update|steady-update|churn|persistent-churn]\n                     [--seed N] [--reps N] [--warmup N]\nMap defaults: keys=65536, batch=keys, queries=65536, capacity=131072,\nscenario=cold-insert, seed=0, reps=30, warmup=5. Counts accept K/M;\nseed accepts decimal or 0x hex. Keys is the live population at lookup;\nbatch is inserted/updated/replaced keys (1 <= batch <= keys).\nCold-insert starts with keys-batch entries; update starts with keys entries;\nchurn erases batch entries and inserts disjoint replacements.\nThose scenarios reset the same baseline outside each sample's timing.\nSteady-update populates once, then repeats the same existing-key/value batch.\nPersistent-churn populates once, then alternates original and disjoint replacement\nbatches, keeping the same live count without resetting through warmup/measurement.\nAll input arrays are prepared before timing. Erase, insert and random hit/miss/mixed\nlookups report separate completed-call CPU-wall latencies, including maintenance.\nSetup, upload/readback/oracle excluded; every result and the full final population\nare checked outside timing. No bandwidth model.");
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
    check_hashmap(&gpu)?;
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

fn map_snapshot(
    map: &mccl::HashMap,
    gpu: &Gpu,
    oracle: &std::collections::HashMap<u32, u32>,
    queries: &[u32],
) -> Result<()> {
    if map.len() != oracle.len() || map.is_empty() != oracle.is_empty() {
        return Err("map host length disagrees with oracle".into());
    }
    let result = map.find(&gpu.upload(queries)?)?;
    let values: Vec<u32> = queries
        .iter()
        .map(|key| oracle.get(key).copied().unwrap_or(0))
        .collect();
    let found: Vec<u32> = queries
        .iter()
        .map(|key| u32::from(oracle.contains_key(key)))
        .collect();
    exact("map values", &result.values.to_vec()?, &values)?;
    exact("map found", &result.found.to_vec()?, &found)
}

fn map_report(
    actual: mccl::InsertReport,
    inserted: usize,
    full: usize,
    invalid: usize,
) -> Result<()> {
    let expected = mccl::InsertReport {
        inserted,
        full,
        invalid,
    };
    if actual != expected {
        return Err(format!("map report {actual:?}, expected {expected:?}").into());
    }
    Ok(())
}

fn reference_map_hash(key: u32, seed: u32) -> u32 {
    let mut hash = seed.wrapping_add(0x1656_67b1).wrapping_add(4);
    hash = hash.wrapping_add(key.wrapping_mul(0xc2b2_ae3d));
    hash = hash.rotate_left(17).wrapping_mul(0x27d4_eb2f);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x85eb_ca77);
    hash ^= hash >> 13;
    hash = hash.wrapping_mul(0xc2b2_ae3d);
    hash ^ (hash >> 16)
}

fn check_map_winner_lifecycle(gpu: &Gpu) -> Result<()> {
    use std::collections::HashMap as Oracle;
    for seed in [0, 1, 0x9e37_79b9, u32::MAX] {
        // Wrap a long collision chain at creation; retain these exact keys as
        // rehashes change the hash modulus and the sparse publication cutoff.
        let colliders: Vec<u32> = (0..u32::MAX - 1)
            .filter(|&key| reference_map_hash(key, seed) % 257 == 256)
            .take(50)
            .collect();
        let keys = &colliders[..48];
        let absent = &colliders[48..];
        let queries: Vec<u32> = colliders
            .iter()
            .copied()
            .chain([u32::MAX - 1, u32::MAX])
            .collect();
        let initial_values: Vec<u32> = (0..keys.len())
            .map(|i| [0, u32::MAX, u32::MAX - 1, 1 << 31][i % 4])
            .collect();
        let initial_input = gpu.upload(keys)?;
        let initial_value_input = gpu.upload(&initial_values)?;
        let mut map = gpu.hash_map(257, seed)?;
        let mut oracle: Oracle<u32, u32> = keys
            .iter()
            .copied()
            .zip(initial_values.iter().copied())
            .collect();
        map_report(map.insert(&initial_input, &initial_value_input)?, 48, 0, 0)?;
        map_snapshot(&map, gpu, &oracle, &queries)?;
        for capacity in [257, 521, 129, 48, 4096, 257] {
            map.rehash(capacity)?;
            if map.capacity() != capacity || map.seed() != seed {
                return Err("winner lifecycle rehash metadata mismatch".into());
            }
            map_snapshot(&map, gpu, &oracle, &queries)?;
            let cutoff = capacity as usize / 8;
            for round in 0..3 {
                for count in [cutoff + 1, 1, cutoff - 1, cutoff, cutoff + 1, cutoff, 1] {
                    let batch: Vec<u32> = (0..count)
                        .map(|i| match i % 13 {
                            11 => u32::MAX,
                            12 => u32::MAX - 1,
                            _ => keys[i % 3],
                        })
                        .collect();
                    let values: Vec<u32> = (0..count)
                        .map(|i| {
                            if round == 1 {
                                // Repeated values must still clear the winning
                                // ordinal before a later, shorter batch.
                                71
                            } else {
                                (capacity << 20)
                                    ^ ((round as u32) << 16)
                                    ^ ((count as u32) << 8)
                                    ^ i as u32
                            }
                        })
                        .collect();
                    let invalid = batch.iter().filter(|&&key| key >= u32::MAX - 1).count();
                    oracle.extend(
                        batch
                            .iter()
                            .copied()
                            .zip(values.iter().copied())
                            .filter(|&(key, _)| key < u32::MAX - 1),
                    );
                    map_report(
                        map.insert(&gpu.upload(&batch)?, &gpu.upload(&values)?)?,
                        0,
                        0,
                        invalid,
                    )?;
                    map_snapshot(&map, gpu, &oracle, &queries)?;
                }
            }
            // Rejected batches must not leave scratch winners that affect a
            // subsequent successful update, including when every slot is full.
            for rejected in [
                vec![u32::MAX, u32::MAX - 1, u32::MAX],
                if capacity == 48 {
                    vec![absent[0], absent[1], absent[0]]
                } else {
                    vec![u32::MAX - 1]
                },
            ] {
                let invalid = rejected.iter().filter(|&&key| key >= u32::MAX - 1).count();
                map_report(
                    map.insert(
                        &gpu.upload(&rejected)?,
                        &gpu.upload(&vec![9; rejected.len()])?,
                    )?,
                    0,
                    rejected.len() - invalid,
                    invalid,
                )?;
                map_snapshot(&map, gpu, &oracle, &queries)?;
                map_report(
                    map.insert(&gpu.upload(&[keys[0]])?, &gpu.upload(&[0])?)?,
                    0,
                    0,
                    0,
                )?;
                oracle.insert(keys[0], 0);
                map_snapshot(&map, gpu, &oracle, &queries)?;
            }
            // Small deletion batches leave a live table with tombstones.
            // Absent-only erases and finds preserve that state; a later insertion
            // reuses holes or performs maintenance according to occupancy.
            let erased = [keys[0], keys[1], keys[2], keys[0], u32::MAX];
            let erase_input = gpu.upload(&erased)?;
            for round in 0..3 {
                if map.erase(&erase_input)? != 3 {
                    return Err("winner lifecycle small erase count mismatch".into());
                }
                for key in &erased {
                    oracle.remove(key);
                }
                map_snapshot(&map, gpu, &oracle, &queries)?;
                for _ in 0..3 {
                    if map.erase(&erase_input)? != 0 {
                        return Err("winner lifecycle absent erase changed count".into());
                    }
                    map_snapshot(&map, gpu, &oracle, &queries)?;
                }
                if round == 1 {
                    map.rehash(capacity)?;
                    map_snapshot(&map, gpu, &oracle, &queries)?;
                }
                let batch = [keys[0], keys[1], keys[2], keys[0], keys[1], keys[2]];
                let values = [0, 1, 2, u32::MAX, 71, round];
                map_report(
                    map.insert(&gpu.upload(&batch)?, &gpu.upload(&values)?)?,
                    3,
                    0,
                    0,
                )?;
                oracle.extend(batch.into_iter().zip(values));
                map_snapshot(&map, gpu, &oracle, &queries)?;
            }
        }
        // Clear and reuse the same map and its grown insertion scratch, first
        // with a large duplicate winner ordinal, then with ordinal one.
        for value in [0, u32::MAX, 71] {
            map.clear()?;
            oracle.clear();
            map_snapshot(&map, gpu, &oracle, &queries)?;
            map_report(
                map.insert(
                    &gpu.upload(&vec![keys[0]; 129])?,
                    &gpu.upload(&vec![value; 129])?,
                )?,
                1,
                0,
                0,
            )?;
            oracle.insert(keys[0], value);
            map_snapshot(&map, gpu, &oracle, &queries)?;
            map_report(
                map.insert(&gpu.upload(&[keys[0]])?, &gpu.upload(&[!value])?)?,
                0,
                0,
                0,
            )?;
            oracle.insert(keys[0], !value);
            map_snapshot(&map, gpu, &oracle, &queries)?;
        }
    }
    Ok(())
}

fn check_map_persistent_churn(gpu: &Gpu) -> Result<()> {
    use std::collections::HashMap as Oracle;
    for seed in [0, 0x9e37_79b9, u32::MAX] {
        for capacity in [64u32, 67] {
            let threshold = (capacity / 16) as usize;
            let high_occupancy = (15 * capacity).div_ceil(16) as usize;
            for live in [
                capacity as usize / 2,
                high_occupancy - 1,
                high_occupancy,
                capacity as usize,
            ] {
                // Adjacent home buckets create overlapping chains that wrap
                // across slot zero. Insert singly so the erased prefix really
                // precedes surviving keys, independent of GPU scheduling.
                let keys: Vec<u32> = (0..u32::MAX - 1)
                    .filter(|&key| reference_map_hash(key, seed) % capacity >= capacity - 2)
                    .take(live * 2 + 1)
                    .collect();
                let mut map = gpu.hash_map(capacity, seed)?;
                let mut oracle = Oracle::new();
                let mut current = keys[..live].to_vec();
                for &key in &current {
                    map_report(
                        map.insert(&gpu.upload(&[key])?, &gpu.upload(&[!key])?)?,
                        1,
                        0,
                        0,
                    )?;
                    oracle.insert(key, !key);
                }
                let queries: Vec<u32> = keys
                    .iter()
                    .copied()
                    .chain([u32::MAX - 1, u32::MAX])
                    .collect();
                let query_input = gpu.upload(&queries)?;
                map_snapshot(&map, gpu, &oracle, &queries)?;
                for round in 0..12 {
                    let saved = map.find(&query_input)?;
                    let saved_values: Vec<u32> = queries
                        .iter()
                        .map(|key| oracle.get(key).copied().unwrap_or(0))
                        .collect();
                    let saved_flags: Vec<u32> = queries
                        .iter()
                        .map(|key| u32::from(oracle.contains_key(key)))
                        .collect();
                    // Exercise both sides of the maintenance tombstone limit,
                    // plus both sides of the occupied-slot limit above.
                    let count = [1, 1, threshold - 1, threshold, threshold + 1, 1][round % 6];
                    let erased: Vec<u32> = current[..count]
                        .iter()
                        .chain(&current[..count])
                        .copied()
                        .chain([keys[live * 2], u32::MAX, u32::MAX - 1])
                        .collect();
                    let erase_input = gpu.upload(&erased)?;
                    if map.erase(&erase_input)? != count {
                        return Err("persistent collision churn erase count mismatch".into());
                    }
                    for key in &erased {
                        oracle.remove(key);
                    }
                    map_snapshot(&map, gpu, &oracle, &queries)?;
                    if map.erase(&erase_input)? != 0 {
                        return Err(
                            "persistent collision churn repeated erase changed length".into()
                        );
                    }
                    map_snapshot(&map, gpu, &oracle, &queries)?;
                    let survivor = current[live - 1];
                    if round % 6 == 0 {
                        // Updating an existing key behind a hole must not claim
                        // the hole; the following missing-only duplicate batch
                        // must claim exactly one slot, even on the dense path.
                        let values = [round as u32, u32::MAX - round as u32];
                        map_report(
                            map.insert(&gpu.upload(&[survivor; 2])?, &gpu.upload(&values)?)?,
                            0,
                            0,
                            0,
                        )?;
                        oracle.insert(survivor, values[1]);
                        map_snapshot(&map, gpu, &oracle, &queries)?;
                    }
                    let replacements: Vec<u32> = (0..count)
                        .map(|i| {
                            if current[i] == keys[i] {
                                keys[live + i]
                            } else {
                                keys[i]
                            }
                        })
                        .collect();
                    let mut batch = if round % 6 == 0 {
                        vec![replacements[0]; 1025]
                    } else {
                        let mut batch = replacements.clone();
                        batch.push(survivor);
                        batch.extend(&replacements);
                        batch.push(survivor);
                        if round >= 6 {
                            // Force dense publication with the same logical set
                            // as the earlier sparse mixed-duplicate batch.
                            batch.extend((0..1025).map(|i| {
                                if i % 2 == 0 {
                                    replacements[0]
                                } else {
                                    survivor
                                }
                            }));
                        }
                        batch
                    };
                    batch.extend([u32::MAX, u32::MAX - 1]);
                    let values: Vec<u32> = (0..batch.len())
                        .map(|i| ((round as u32) << 24) ^ i as u32)
                        .collect();
                    map_report(
                        map.insert(&gpu.upload(&batch)?, &gpu.upload(&values)?)?,
                        count,
                        0,
                        2,
                    )?;
                    oracle.extend(
                        batch
                            .iter()
                            .copied()
                            .zip(values)
                            .filter(|&(key, _)| key < u32::MAX - 1),
                    );
                    current[..count].copy_from_slice(&replacements);
                    map_snapshot(&map, gpu, &oracle, &queries)?;
                    if map.len() != live {
                        return Err("persistent collision churn changed live population".into());
                    }
                    if live == capacity as usize {
                        let rejected = [keys[live * 2], survivor, keys[live * 2], u32::MAX];
                        let values = [1, round as u32, 2, 3];
                        map_report(
                            map.insert(&gpu.upload(&rejected)?, &gpu.upload(&values)?)?,
                            0,
                            2,
                            1,
                        )?;
                        oracle.insert(survivor, round as u32);
                        map_snapshot(&map, gpu, &oracle, &queries)?;
                    }
                    if round == 5 {
                        map.rehash(capacity)?;
                        map_snapshot(&map, gpu, &oracle, &queries)?;
                    } else if round == 9 {
                        map.rehash(capacity * 2 + 1)?;
                        map_snapshot(&map, gpu, &oracle, &queries)?;
                        map.rehash(capacity)?;
                        map_snapshot(&map, gpu, &oracle, &queries)?;
                    }
                    exact(
                        "persistent churn retained values",
                        &saved.values.to_vec()?,
                        &saved_values,
                    )?;
                    exact(
                        "persistent churn retained flags",
                        &saved.found.to_vec()?,
                        &saved_flags,
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn check_hashmap(gpu: &Gpu) -> Result<()> {
    use std::collections::HashMap as Oracle;
    for capacity in [0, i32::MAX as u32 + 1, u32::MAX] {
        if gpu.hash_map(capacity, 0).is_ok() {
            return Err("invalid map capacity accepted".into());
        }
    }
    let empty = gpu.upload(&[])?;
    for seed in [0, 1, u32::MAX] {
        for capacity in [1, 3, 17, 257] {
            let mut map = gpu.hash_map(capacity, seed)?;
            let mut oracle = Oracle::new();
            if map.capacity() != capacity || map.seed() != seed || !map.is_empty() {
                return Err("incorrect new map metadata".into());
            }
            map_report(map.insert(&empty, &empty)?, 0, 0, 0)?;
            if map.erase(&empty)? != 0 {
                return Err("empty erase was nonzero".into());
            }
            map_snapshot(&map, gpu, &oracle, &[])?;
            map_snapshot(&map, gpu, &oracle, &[0, u32::MAX, u32::MAX - 1])?;
            // Force one long wrapping collision chain, including non-power-of-two capacities.
            let keys: Vec<u32> = (0..u32::MAX - 1)
                .filter(|&key| reference_map_hash(key, seed) % capacity == capacity - 1)
                .take(capacity as usize)
                .collect();
            let values: Vec<u32> = keys
                .iter()
                .enumerate()
                .map(|(i, _)| [0, u32::MAX, u32::MAX - 1, 1 << 31][i % 4])
                .collect();
            // Leave one slot vacant, then fill it through a duplicate-heavy batch.
            let n = keys.len() - 1;
            map_report(
                map.insert(&gpu.upload(&keys[..n])?, &gpu.upload(&values[..n])?)?,
                n,
                0,
                0,
            )?;
            oracle.extend(keys[..n].iter().copied().zip(values[..n].iter().copied()));
            map_snapshot(&map, gpu, &oracle, &keys)?;
            let duplicate_keys = vec![keys[n]; 1025];
            let duplicate_values: Vec<u32> = (0..1025).collect();
            map_report(
                map.insert(
                    &gpu.upload(&duplicate_keys)?,
                    &gpu.upload(&duplicate_values)?,
                )?,
                1,
                0,
                0,
            )?;
            oracle.insert(keys[n], 1024);
            map_snapshot(&map, gpu, &oracle, &keys)?;
            // Full maps must still find/update existing entries, and count invalid
            // elements separately from valid inputs that cannot be admitted.
            let absent = (0..u32::MAX - 1)
                .find(|key| !oracle.contains_key(key))
                .unwrap();
            let mixed = [u32::MAX, keys[0], absent, u32::MAX - 1, keys[0], absent];
            let mixed_values = [7, 8, 9, 10, u32::MAX, 12];
            map_report(
                map.insert(&gpu.upload(&mixed)?, &gpu.upload(&mixed_values)?)?,
                0,
                2,
                2,
            )?;
            oracle.insert(keys[0], u32::MAX);
            map_snapshot(&map, gpu, &oracle, &mixed)?;
            let all_updates: Vec<u32> = keys.iter().chain(keys.iter()).copied().collect();
            let all_values: Vec<u32> = (0..all_updates.len() as u32).collect();
            map_report(
                map.insert(&gpu.upload(&all_updates)?, &gpu.upload(&all_values)?)?,
                0,
                0,
                0,
            )?;
            oracle.extend(all_updates.iter().copied().zip(all_values));
            let snapshot = map.find(&gpu.upload(&keys)?)?;
            let saved: Vec<u32> = keys.iter().map(|k| oracle[k]).collect();
            for bad in [0, capacity - 1, u32::MAX] {
                if map.rehash(bad).is_ok() || map.capacity() != capacity {
                    return Err("invalid shrink changed map capacity".into());
                }
                map_snapshot(&map, gpu, &oracle, &keys)?;
            }
            map.rehash(capacity * 2 + 1)?;
            map_snapshot(&map, gpu, &oracle, &keys)?;
            map.rehash(capacity)?;
            map_snapshot(&map, gpu, &oracle, &keys)?;
            let erased_keys = [keys[0], keys[0], absent, u32::MAX, u32::MAX - 1];
            if map.erase(&gpu.upload(&erased_keys)?)? != 1 {
                return Err("duplicate erase was not counted uniquely".into());
            }
            oracle.remove(&keys[0]);
            map_snapshot(&map, gpu, &oracle, &keys)?;
            // Existing colliders beyond a tombstone must not gain duplicate slots.
            let replacement: Vec<u32> = keys.iter().map(|_| 71).collect();
            map_report(
                map.insert(&gpu.upload(&keys)?, &gpu.upload(&replacement)?)?,
                1,
                0,
                0,
            )?;
            oracle.extend(keys.iter().copied().map(|key| (key, 71)));
            map_snapshot(&map, gpu, &oracle, &keys)?;
            map.clear()?;
            oracle.clear();
            map_snapshot(&map, gpu, &oracle, &keys)?;
            exact(
                "map pre-mutation snapshot values",
                &snapshot.values.to_vec()?,
                &saved,
            )?;
            exact(
                "map pre-mutation snapshot found",
                &snapshot.found.to_vec()?,
                &vec![1; keys.len()],
            )?;
            map_report(
                map.insert(
                    &gpu.upload(&[u32::MAX, 0, u32::MAX - 1, 0])?,
                    &gpu.upload(&[1, 2, 3, 4])?,
                )?,
                1,
                0,
                2,
            )?;
            oracle.insert(0, 4);
            map_snapshot(&map, gpu, &oracle, &[0, u32::MAX, u32::MAX - 1])?;
        }
    }
    // Reuse insertion scratch across large, tiny, then larger batches. Include
    // invalid lanes and repeated keys so stale slots or winner ordinals cannot
    // silently overwrite a valid value. The final phase starts from a full map.
    let mut resized = gpu.hash_map(37, 123)?;
    let mut resized_oracle = Oracle::new();
    for (batch_len, key_space) in [(513, 31), (1, 31), (1537, 37), (3, 43), (2049, 43)] {
        let keys: Vec<u32> = (0..batch_len)
            .map(|i| match i % 29 {
                27 => u32::MAX,
                28 => u32::MAX - 1,
                _ => (i % key_space) as u32,
            })
            .collect();
        let values: Vec<u32> = (0..batch_len)
            .map(|i| (batch_len as u32).rotate_left(17) ^ i as u32)
            .collect();
        let before = resized_oracle.len();
        let mut full = 0;
        let mut invalid = 0;
        for (&key, &value) in keys.iter().zip(&values) {
            if key >= u32::MAX - 1 {
                invalid += 1;
            } else if key >= 37 {
                full += 1;
            } else {
                resized_oracle.insert(key, value);
            }
        }
        map_report(
            resized.insert(&gpu.upload(&keys)?, &gpu.upload(&values)?)?,
            resized_oracle.len() - before,
            full,
            invalid,
        )?;
        let queries: Vec<u32> = (0..43).chain([u32::MAX - 1, u32::MAX]).collect();
        map_snapshot(&resized, gpu, &resized_oracle, &queries)?;
    }
    // Erase-all leaves only tombstones; repeatedly refill with different keys
    // and preserve an earlier eager lookup snapshot across all maintenance.
    let original_keys: Vec<u32> = (0..37).collect();
    let snapshot = resized.find(&gpu.upload(&original_keys)?)?;
    let saved: Vec<u32> = original_keys
        .iter()
        .map(|key| resized_oracle[key])
        .collect();
    let mut current_keys = original_keys.clone();
    for round in 1..=4 {
        let duplicate_erases: Vec<u32> = current_keys
            .iter()
            .chain(&current_keys)
            .copied()
            .chain([u32::MAX, u32::MAX - 1])
            .collect();
        if resized.erase(&gpu.upload(&duplicate_erases)?)? != 37 || !resized.is_empty() {
            return Err("erase-all duplicate count mismatch".into());
        }
        resized_oracle.clear();
        map_snapshot(&resized, gpu, &resized_oracle, &current_keys)?;
        current_keys = (round * 37..(round + 1) * 37).collect();
        let duplicate_inserts: Vec<u32> =
            current_keys.iter().chain(&current_keys).copied().collect();
        let values: Vec<u32> = (0..duplicate_inserts.len() as u32).collect();
        map_report(
            resized.insert(&gpu.upload(&duplicate_inserts)?, &gpu.upload(&values)?)?,
            37,
            0,
            0,
        )?;
        resized_oracle.extend(duplicate_inserts.into_iter().zip(values));
        let queries: Vec<u32> = original_keys.iter().chain(&current_keys).copied().collect();
        map_snapshot(&resized, gpu, &resized_oracle, &queries)?;
    }
    exact(
        "erase-all retained snapshot values",
        &snapshot.values.to_vec()?,
        &saved,
    )?;
    exact(
        "erase-all retained snapshot flags",
        &snapshot.found.to_vec()?,
        &[1; 37],
    )?;
    // Random ordered operations remain below capacity, so the CPU oracle can
    // assert every admission as well as deterministic last-input updates.
    let mut random = Random(0x1234_5678_abcd_ef01);
    let mut map = gpu.hash_map(131, 42)?;
    let mut oracle = Oracle::new();
    let queries: Vec<u32> = (0..129).chain([u32::MAX - 1, u32::MAX]).collect();
    for round in 0..64 {
        let keys: Vec<u32> = (0..193).map(|_| random.next() % 129).collect();
        if round % 3 == 0 {
            let before = oracle.len();
            for key in &keys {
                oracle.remove(key);
            }
            if map.erase(&gpu.upload(&keys)?)? != before - oracle.len() {
                return Err("random erase count disagrees with oracle".into());
            }
        } else {
            let values: Vec<u32> = keys.iter().map(|_| random.next()).collect();
            let before = oracle.len();
            oracle.extend(keys.iter().copied().zip(values.iter().copied()));
            map_report(
                map.insert(&gpu.upload(&keys)?, &gpu.upload(&values)?)?,
                oracle.len() - before,
                0,
                0,
            )?;
        }
        map_snapshot(&map, gpu, &oracle, &queries)?;
    }
    // Over-capacity admission is scheduling-dependent; validate the selected
    // subset against the oracle rather than assuming which keys win.
    let mut small = gpu.hash_map(3, 7)?;
    let keys: Vec<u32> = (0..19).collect();
    let values: Vec<u32> = keys.iter().map(|key| key + 100).collect();
    map_report(
        small.insert(&gpu.upload(&keys)?, &gpu.upload(&values)?)?,
        3,
        16,
        0,
    )?;
    let selected = small.find(&gpu.upload(&keys)?)?;
    let flags = selected.found.to_vec()?;
    let actual = selected.values.to_vec()?;
    if small.len() != 3 || flags.iter().sum::<u32>() != 3 {
        return Err("over-capacity insertion occupancy mismatch".into());
    }
    for i in 0..keys.len() {
        if flags[i] > 1 || actual[i] != if flags[i] == 1 { values[i] } else { 0 } {
            return Err("over-capacity insertion value mismatch".into());
        }
    }
    let other = Gpu::new()?;
    let foreign = other.upload(&[0])?;
    let local = gpu.upload(&[0])?;
    let before = map.len();
    if map.insert(&foreign, &local).is_ok()
        || map.insert(&local, &foreign).is_ok()
        || map.find(&foreign).is_ok()
        || map.erase(&foreign).is_ok()
        || map.insert(&local, &empty).is_ok()
        || map.len() != before
    {
        return Err("map input validation failed or changed host length".into());
    }
    map_snapshot(&map, gpu, &oracle, &queries)?;
    let (mut retained, input, values) = {
        let context = Gpu::new()?;
        (
            context.hash_map(3, 99)?,
            context.upload(&[0, 1])?,
            context.upload(&[u32::MAX, 0])?,
        )
    };
    map_report(retained.insert(&input, &values)?, 2, 0, 0)?;
    let output = retained.find(&input)?;
    retained.clear()?;
    drop(retained);
    drop(input);
    drop(values);
    exact(
        "retained map values",
        &output.values.to_vec()?,
        &[u32::MAX, 0],
    )?;
    exact("retained map flags", &output.found.to_vec()?, &[1, 1])?;
    check_map_winner_lifecycle(gpu)?;
    check_map_persistent_churn(gpu)?;
    // Full tables force bounded wraparound misses; these capacities exercise
    // scalar and 4/8/16-slot specializations, including partial-vector origins.
    for capacity in [3_u32, 4, 8, 12, 16, 32, 36] {
        let mut table = gpu.hash_map(capacity, 57)?;
        let keys: Vec<u32> = (0..capacity).collect();
        let values: Vec<u32> = keys.iter().map(|k| !k).collect();
        let mut expected: Oracle<u32, u32> =
            keys.iter().copied().zip(values.iter().copied()).collect();
        map_report(
            table.insert(&gpu.upload(&keys)?, &gpu.upload(&values)?)?,
            capacity as usize,
            0,
            0,
        )?;
        let queries: Vec<u32> = (0..capacity + 2).collect();
        map_snapshot(&table, gpu, &expected, &queries)?;
        let erased = [0, capacity - 1];
        if table.erase(&gpu.upload(&erased)?)? != 2 {
            return Err("lookup-width erase mismatch".into());
        }
        for key in erased {
            expected.remove(&key);
        }
        map_snapshot(&table, gpu, &expected, &queries)?;
        table.clear()?;
        expected.clear();
        map_snapshot(&table, gpu, &expected, &queries)?;
    }
    println!("PASS: GPU map CPU oracle, duplicate updates, collision chains, full/partial admission,\n reserved keys, all value bits, erase/reinsert, rehash preservation, snapshots and context lifetime,\n reused winner lifecycle across sparse/dense cutoffs, tombstones, clear and capacity changes,\n persistent overlapping/wrapping collision churn across maintenance thresholds");
    Ok(())
}

struct MapBenchOptions {
    keys: usize,
    batch: usize,
    queries: usize,
    capacity: u32,
    scenario: &'static str,
    seed: u32,
    reps: usize,
    warmup: usize,
}

impl MapBenchOptions {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self> {
        let mut options = Self {
            keys: 65536,
            batch: 0,
            queries: 65536,
            capacity: 131072,
            scenario: "cold-insert",
            seed: 0,
            reps: 30,
            warmup: 5,
        };
        let mut batch = None;
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--keys" => options.keys = parse_size(&value)?,
                "--batch" => batch = Some(parse_size(&value)?),
                "--queries" => options.queries = parse_size(&value)?,
                "--capacity" => options.capacity = u32::try_from(parse_size(&value)?)?,
                "--scenario" => {
                    options.scenario = match value.as_str() {
                        "cold-insert" => "cold-insert",
                        "update" => "update",
                        "steady-update" => "steady-update",
                        "churn" => "churn",
                        "persistent-churn" => "persistent-churn",
                        _ => return Err(format!("unknown map benchmark scenario {value:?}").into()),
                    };
                }
                "--seed" => {
                    options.seed = if let Some(hex) = value.strip_prefix("0x") {
                        u32::from_str_radix(hex, 16)?
                    } else {
                        value.parse()?
                    }
                }
                "--reps" => options.reps = value.parse()?,
                "--warmup" => options.warmup = value.parse()?,
                _ => return Err(format!("unknown map benchmark option {flag:?}").into()),
            }
        }
        options.batch = batch.unwrap_or(options.keys);
        if options.capacity == 0
            || options.capacity > i32::MAX as u32
            || options.keys == 0
            || options.keys > options.capacity as usize
            || options.batch == 0
            || options.batch > options.keys
            || options.queries < 2
            || options.queries > i32::MAX as usize
            || options.reps == 0
            || options.warmup == 0
        {
            return Err("map benchmark needs 1 <= batch <= keys <= capacity <= INT_MAX, 2 <= queries <= INT_MAX, and nonzero reps/warmup".into());
        }
        Ok(options)
    }
}

fn map_bench_random_below(random: &mut Random, bound: u32) -> u32 {
    // Rejection sampling avoids modulo bias, including for the large absent
    // key domain. Callers always supply a nonzero bound.
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let value = random.next();
        if value >= threshold {
            return value % bound;
        }
    }
}

fn map_bench_shuffle(keys: &mut [u32], random: &mut Random) {
    for i in (1..keys.len()).rev() {
        let j = map_bench_random_below(random, (i + 1) as u32) as usize;
        keys.swap(i, j);
    }
}

fn hashmap_bench(options: MapBenchOptions) -> Result<()> {
    use std::collections::HashMap as Oracle;
    let gpu = Gpu::new()?;
    let mut map = gpu.hash_map(options.capacity, options.seed)?;
    let mut random = Random(0x1234_5678_abcd_ef01 ^ u64::from(options.seed));
    let key_count = options.keys as u32;
    let batch_count = options.batch as u32;
    let persistent = options.scenario == "persistent-churn";
    let churn = options.scenario == "churn" || persistent;
    let steady = options.scenario == "steady-update";
    let update = options.scenario == "update" || steady;
    let baseline_start = if options.scenario == "cold-insert" {
        batch_count
    } else {
        0
    };
    let mut baseline_keys: Vec<u32> = (baseline_start..key_count).collect();
    map_bench_shuffle(&mut baseline_keys, &mut random);
    let baseline_values: Vec<u32> = baseline_keys.iter().map(|key| !key).collect();
    let baseline_input = gpu.upload(&baseline_keys)?;
    let baseline_value_input = gpu.upload(&baseline_values)?;
    struct Phase {
        input: U32Array,
        values: U32Array,
        erase: Option<U32Array>,
        lookups: Vec<(U32Array, Vec<u32>, Vec<u32>)>,
        baseline_values: Vec<u32>,
        baseline_flags: Vec<u32>,
        batch_values: Vec<u32>,
    }
    let patterns = ["random-hit", "random-miss", "mixed-50"];
    let mut phases = Vec::new();
    // Both directions, including query arrays and CPU expectations, are resident
    // before warmup. Only persistent churn alternates phases without resetting.
    for phase in 0..if persistent { 2 } else { 1 } {
        let replacing = churn && phase == 0;
        let mut batch_keys: Vec<u32> = if replacing {
            (key_count..key_count + batch_count).collect()
        } else {
            (0..batch_count).collect()
        };
        map_bench_shuffle(&mut batch_keys, &mut random);
        let batch_values: Vec<u32> = batch_keys
            .iter()
            .map(|&key| {
                if update || phase == 1 {
                    key ^ 0xa5a5_a5a5
                } else {
                    !key
                }
            })
            .collect();
        let erase = if churn {
            let mut keys: Vec<u32> = if replacing {
                (0..batch_count).collect()
            } else {
                (key_count..key_count + batch_count).collect()
            };
            map_bench_shuffle(&mut keys, &mut random);
            Some(gpu.upload(&keys)?)
        } else {
            None
        };
        let mut oracle: Oracle<u32, u32> = baseline_keys
            .iter()
            .copied()
            .zip(baseline_values.iter().copied())
            .collect();
        if replacing {
            for key in 0..batch_count {
                oracle.remove(&key);
            }
        }
        oracle.extend(batch_keys.iter().copied().zip(batch_values.iter().copied()));
        // Live keys form one contiguous range in each phase. Map uniform absent
        // ranks around it, excluding reserved keys even at maximum capacity.
        let live_start = if replacing { batch_count } else { 0 };
        let absent_count = (u32::MAX - 1) - key_count;
        let mut lookups = Vec::new();
        for pattern in patterns {
            let mut queries: Vec<u32> = (0..options.queries)
                .map(|i| {
                    if pattern == "random-hit" || (pattern == "mixed-50" && i % 2 == 0) {
                        live_start + map_bench_random_below(&mut random, key_count)
                    } else {
                        let rank = map_bench_random_below(&mut random, absent_count);
                        if rank < live_start {
                            rank
                        } else {
                            rank + key_count
                        }
                    }
                })
                .collect();
            map_bench_shuffle(&mut queries, &mut random);
            let expected_values = queries
                .iter()
                .map(|key| oracle.get(key).copied().unwrap_or(0))
                .collect();
            let expected_flags = queries
                .iter()
                .map(|key| u32::from(oracle.contains_key(key)))
                .collect();
            lookups.push((gpu.upload(&queries)?, expected_values, expected_flags));
        }
        phases.push(Phase {
            input: gpu.upload(&batch_keys)?,
            values: gpu.upload(&batch_values)?,
            erase,
            lookups,
            baseline_values: baseline_keys
                .iter()
                .map(|key| oracle.get(key).copied().unwrap_or(0))
                .collect(),
            baseline_flags: baseline_keys
                .iter()
                .map(|key| u32::from(oracle.contains_key(key)))
                .collect(),
            batch_values,
        });
    }
    let batch_expected_flags = vec![1; options.batch];
    let erased_expected = vec![0; options.batch];
    let mut lookup_samples: Vec<_> = patterns
        .iter()
        .map(|_| Vec::with_capacity(options.reps))
        .collect();
    eprintln!("Map timing: CPU-wall completed calls, including GPU maintenance; inputs resident.");
    if steady {
        eprintln!("Steady-update populates once before warmup; no reset/repopulation between samples.\nEvery insertion repeats the same existing-key/value batch, so measured logical\nstate is unchanged after warmup, but each call must execute and return exact counters.");
    } else if persistent {
        eprintln!("Persistent-churn populates once before warmup, then alternates original and\ndisjoint replacement batches without resetting. Erase, insertion (including any\nmaintenance) and each lookup are timed separately; live population stays fixed.");
    } else {
        eprintln!("Each sample clears/repopulates the same baseline outside timing.\nChurn times erase then replacement insert, including any tombstone rebuild;\nreplacement keys differ from erased keys, so churn changes the logical state.");
    }
    eprintln!("Three lookup patterns see the final population: uniform hits, uniform nonreserved\nabsent keys, and shuffled 50/50 (one extra hit for odd query counts).\nCreation, initial setup, reset, uploads, readbacks, CPU oracle and output destruction excluded.\nEvery warmup/sample validates exact counters, all lookup outputs and the full final table.\nNo bandwidth model.");
    let mut inserts = Vec::with_capacity(options.reps);
    let mut erases = Vec::with_capacity(options.reps);
    if steady || persistent {
        map_report(
            map.insert(&baseline_input, &baseline_value_input)?,
            baseline_keys.len(),
            0,
            0,
        )?;
    }
    for sample in 0..options.warmup + options.reps {
        let measured = sample >= options.warmup;
        let phase = &phases[sample % phases.len()];
        if !steady && !persistent {
            map.clear()?;
            map_report(
                map.insert(&baseline_input, &baseline_value_input)?,
                baseline_keys.len(),
                0,
                0,
            )?;
        }
        if let Some(erase_input) = &phase.erase {
            let start = Instant::now();
            let erased = map.erase(erase_input)?;
            let elapsed = start.elapsed().as_secs_f64();
            if measured {
                erases.push(elapsed);
            }
            if erased != options.batch || map.len() != options.keys - options.batch {
                return Err("map benchmark erase count mismatch".into());
            }
        }
        let start = Instant::now();
        let report = map.insert(&phase.input, &phase.values)?;
        let elapsed = start.elapsed().as_secs_f64();
        if measured {
            inserts.push(elapsed);
        }
        map_report(report, if update { 0 } else { options.batch }, 0, 0)?;
        if map.len() != options.keys {
            return Err("map benchmark live count mismatch".into());
        }
        for ((queries, expected_values, expected_flags), samples) in
            phase.lookups.iter().zip(&mut lookup_samples)
        {
            let start = Instant::now();
            let output = map.find(queries)?;
            let elapsed = start.elapsed().as_secs_f64();
            if measured {
                samples.push(elapsed);
            }
            exact(
                "map benchmark values",
                &output.values.to_vec()?,
                expected_values,
            )?;
            exact(
                "map benchmark found",
                &output.found.to_vec()?,
                expected_flags,
            )?;
        }
        for (queries, values, flags) in [
            (
                &baseline_input,
                &phase.baseline_values,
                &phase.baseline_flags,
            ),
            (&phase.input, &phase.batch_values, &batch_expected_flags),
        ] {
            let output = map.find(queries)?;
            exact(
                "map benchmark full values",
                &output.values.to_vec()?,
                values,
            )?;
            exact("map benchmark full found", &output.found.to_vec()?, flags)?;
        }
        if let Some(erased) = &phase.erase {
            let output = map.find(erased)?;
            exact(
                "map benchmark erased values",
                &output.values.to_vec()?,
                &erased_expected,
            )?;
            exact(
                "map benchmark erased flags",
                &output.found.to_vec()?,
                &erased_expected,
            )?;
        }
    }
    let stdout = io::stdout();
    let mut csv = io::BufWriter::new(stdout.lock());
    writeln!(
        csv,
        "scenario,operation,query_pattern,live_keys,batch_keys,query_keys,capacity,load_factor,seed,reps,warmup,timing,p50_us,p10_us,p90_us"
    )?;
    let mut rows = vec![("insert", "none", inserts)];
    if churn {
        rows.push(("erase", "none", erases));
    }
    rows.extend(
        patterns
            .into_iter()
            .zip(lookup_samples)
            .map(|(pattern, samples)| ("find", pattern, samples)),
    );
    for (operation, pattern, samples) in rows {
        let stats = Stats::new(samples);
        writeln!(
            csv,
            "{},{operation},{pattern},{},{},{},{},{:.6},{},{},{},cpu_wall_completed_call,{:.3},{:.3},{:.3}",
            options.scenario,
            options.keys,
            options.batch,
            options.queries,
            options.capacity,
            options.keys as f64 / f64::from(options.capacity),
            options.seed,
            options.reps,
            options.warmup,
            stats.median * 1e6,
            stats.p10 * 1e6,
            stats.p90 * 1e6
        )?;
    }
    csv.flush()?;
    Ok(())
}
