/// Requires the same built native libraries and Metal GPU as the executable.
/// Kept as one test so GPU stream-affine work stays in one process/thread, rather
/// than Rust's parallel test workers racing individual device test cases.
#[test]
fn exact_gpu_operations_and_bloom_cli_regressions() {
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_mccl"))
        .arg("check")
        .output()
        .expect("launch mccl check");
    assert!(
        result.status.success(),
        "mccl check failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr),
    );
    // Exercise the isolated benchmark path with non-power-of-two block sizing,
    // extreme seed, asymmetric counts, and the supported hexadecimal seed form.
    let benchmark = std::process::Command::new(env!("CARGO_BIN_EXE_mccl"))
        .args([
            "bloom-bench",
            "--keys",
            "257",
            "--queries",
            "513",
            "--blocks",
            "13",
            "--seed",
            "0xffffffffffffffff",
            "--reps",
            "1",
            "--warmup",
            "1",
        ])
        .output()
        .expect("launch Bloom benchmark");
    assert!(
        benchmark.status.success(),
        "Bloom benchmark failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&benchmark.stdout),
        String::from_utf8_lossy(&benchmark.stderr),
    );
    for arguments in [
        vec!["bloom-bench", "--blocks", "0"],
        vec!["bloom-bench", "--blocks", "268435456"],
        vec!["bloom-bench", "--keys", "0"],
        vec!["bloom-bench", "--queries", "1"],
        vec!["bloom-bench", "--reps", "0"],
        vec!["bloom-bench", "--warmup", "0"],
        vec!["bloom-bench", "--seed", "18446744073709551616"],
        vec!["bloom-bench", "--unknown", "1"],
        vec!["bloom-bench", "--blocks"],
        vec!["bloom-check", "--unknown"],
    ] {
        let rejected = std::process::Command::new(env!("CARGO_BIN_EXE_mccl"))
            .args(&arguments)
            .output()
            .expect("launch invalid Bloom command");
        assert!(
            !rejected.status.success(),
            "accepted invalid arguments: {arguments:?}"
        );
    }
}
