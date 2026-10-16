/// Requires the same built native libraries and Metal GPU as the executable.
/// Kept as one test so GPU stream-affine work stays in one process/thread, rather
/// than Rust's parallel test workers racing individual device test cases.
#[test]
fn exact_gpu_merge_and_forced_copy_regressions() {
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
}
