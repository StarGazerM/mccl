/// Requires the same built native libraries and Metal GPU as the executable.
/// Kept as one test so GPU stream-affine work stays in one process/thread, rather
/// than Rust's parallel test workers racing individual device test cases.
#[test]
fn exact_gpu_operations_and_cli_regressions() {
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
        vec!["hashmap-check"],
        vec![
            "hashmap-bench",
            "--keys",
            "17",
            "--queries",
            "35",
            "--capacity",
            "17",
            "--seed",
            "0xffffffff",
            "--reps",
            "2",
            "--warmup",
            "1",
        ],
        vec![
            "hashmap-bench",
            "--keys",
            "1",
            "--queries",
            "2",
            "--capacity",
            "3",
            "--reps",
            "1",
            "--warmup",
            "1",
        ],
    ] {
        let result = std::process::Command::new(env!("CARGO_BIN_EXE_mccl"))
            .args(&arguments)
            .output()
            .expect("launch map command");
        assert!(
            result.status.success(),
            "map command {arguments:?} failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr),
        );
    }
    // Persistent churn never resets: small batches stay below the tombstone
    // threshold, while larger batches and near/full occupancy trigger maintenance.
    // Every cycle checks counters, all values and erased keys outside timing.
    for (scenario, batch, reps) in [
        ("cold-insert", "7", "2"),
        ("update", "7", "2"),
        ("churn", "7", "2"),
        ("persistent-churn", "1", "12"),
        ("persistent-churn", "7", "12"),
        ("persistent-churn", "19", "12"),
    ] {
        let populations = if scenario == "persistent-churn" {
            vec!["19", "33", "35", "37"]
        } else {
            vec!["19", "28", "33"]
        };
        for keys in populations {
            let result = std::process::Command::new(env!("CARGO_BIN_EXE_mccl"))
                .args([
                    "hashmap-bench",
                    "--scenario",
                    scenario,
                    "--capacity",
                    "37",
                    "--keys",
                    keys,
                    "--batch",
                    batch,
                    "--queries",
                    "129",
                    "--seed",
                    "0x12345678",
                    "--reps",
                    reps,
                    "--warmup",
                    "1",
                ])
                .output()
                .expect("launch sparse map benchmark");
            assert!(
                result.status.success(),
                "map {scenario} benchmark with {keys} keys failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr),
            );
            let csv = String::from_utf8(result.stdout).expect("benchmark CSV is UTF-8");
            let mut lines = csv.lines();
            let header: Vec<_> = lines.next().expect("CSV header").split(',').collect();
            let column = |name: &str| header.iter().position(|&field| field == name).unwrap();
            let rows: Vec<Vec<_>> = lines.map(|line| line.split(',').collect()).collect();
            let churn = scenario == "churn" || scenario == "persistent-churn";
            assert_eq!(rows.len(), if churn { 5 } else { 4 });
            let expected_load = keys.parse::<f64>().unwrap() / 37.0;
            let mut patterns = Vec::new();
            let mut operations = Vec::new();
            for row in &rows {
                assert_eq!(row.len(), header.len());
                assert_eq!(row[column("scenario")], scenario);
                assert_eq!(row[column("live_keys")], keys);
                assert_eq!(row[column("batch_keys")], batch);
                assert_eq!(row[column("reps")], reps);
                let load = row[column("load_factor")].parse::<f64>().unwrap();
                assert!((load - expected_load).abs() <= 0.000001);
                let p10 = row[column("p10_us")].parse::<f64>().unwrap();
                let p50 = row[column("p50_us")].parse::<f64>().unwrap();
                let p90 = row[column("p90_us")].parse::<f64>().unwrap();
                assert!(
                    p10.is_finite() && p10 >= 0.0 && p10 <= p50 && p50 <= p90 && p90.is_finite()
                );
                if row[column("operation")] == "find" {
                    patterns.push(row[column("query_pattern")]);
                } else {
                    operations.push(row[column("operation")]);
                }
            }
            assert_eq!(patterns, ["random-hit", "random-miss", "mixed-50"]);
            assert_eq!(
                operations,
                if churn {
                    vec!["insert", "erase"]
                } else {
                    vec!["insert"]
                }
            );
        }
    }
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
        vec!["hashmap-check", "--unknown"],
        vec!["hashmap-bench", "--capacity", "0"],
        vec!["hashmap-bench", "--capacity", "2147483648"],
        vec!["hashmap-bench", "--capacity", "1", "--keys", "2"],
        vec!["hashmap-bench", "--keys", "0"],
        vec!["hashmap-bench", "--batch", "0"],
        vec!["hashmap-bench", "--keys", "7", "--batch", "8"],
        vec!["hashmap-bench", "--scenario", "unknown"],
        vec!["hashmap-bench", "--queries", "1"],
        vec!["hashmap-bench", "--queries", "2147483648"],
        vec!["hashmap-bench", "--seed", "4294967296"],
        vec!["hashmap-bench", "--reps", "0"],
        vec!["hashmap-bench", "--warmup", "0"],
        vec!["hashmap-bench", "--unknown", "1"],
        vec!["hashmap-bench", "--capacity"],
        vec!["hashmap-bench", "--batch"],
        vec!["hashmap-bench", "--scenario"],
    ] {
        let rejected = std::process::Command::new(env!("CARGO_BIN_EXE_mccl"))
            .args(&arguments)
            .output()
            .expect("launch invalid command");
        assert!(
            !rejected.status.success(),
            "accepted invalid arguments: {arguments:?}"
        );
    }
}
