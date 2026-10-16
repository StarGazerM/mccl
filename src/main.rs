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
        Some("help" | "--help" | "-h") | None => {
            println!("Usage:\n  mccl check\n  mccl bench [--size N | --sizes N,N,...] [--reps N] [--warmup N]\n             [--case all|disjoint|reverse-disjoint|interleaved|duplicates]\n\nSizes are total keys (A+B); suffixes K/M mean powers of 1024.\nDefaults: sizes=1024,65536,1048576,16777216, reps=30, warmup=5.\nBenchmark CSV goes to stdout; timing is CPU wall time through completed eval+sync,\nnot GPU timestamps. Inputs are resident; every sample creates a fresh lazy output.\nNative libraries must be built separately with CMake before Cargo.");
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
