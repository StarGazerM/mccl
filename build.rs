use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=MCCL_NATIVE_DIR");
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let configured = env::var_os("MCCL_NATIVE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("build/native"));
    let native = if configured.is_absolute() {
        configured
    } else {
        root.join(configured)
    };
    println!("cargo:rustc-link-search=native={}", native.display());
    println!("cargo:rustc-link-lib=dylib=mccl_native");
    println!("cargo:rustc-link-lib=dylib=mlxc");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", native.display());
    // CMake owns dependency fetching and native compilation, never Cargo.
    for name in ["libmccl_native.dylib", "libmlxc.dylib", "libmlx.dylib"] {
        println!("cargo:rerun-if-changed={}", native.join(name).display());
    }
}
