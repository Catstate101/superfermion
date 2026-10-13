//! Build script for sf-cat's OPTIONAL C++ kernel lane (additive experiment).
//!
//! Compiles `csrc/transpose.cpp` with a local MinGW-w64 g++ when one is
//! found (env `SF_CAT_CXX` = directory containing the toolchain, or the
//! MSYS2 default `C:\msys64\mingw64\bin`), archives it into
//! `libsf_cat_cxx.a` and emits `sf_cat_cxx` for `src/kernels.rs`.
//!
//! When no compiler is found (or compilation fails) the kernels are simply
//! not built: the crate remains fully functional and uses its Rust scalar
//! path everywhere.
//!
//! MSYS2 note: `g++.exe` sits in `mingw64/bin` next to its DLLs, but the
//! real compiler `cc1plus.exe` lives under `lib/gcc/...` and cannot start
//! without that `bin` directory on `PATH` — it dies with a silent exit 1.
//! Both spawns below therefore prepend the toolchain directory to `PATH`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn find_tool(dir: Option<&str>, exe: &str) -> Option<PathBuf> {
    if let Some(dir) = dir {
        let p = Path::new(dir).join(exe);
        if p.exists() {
            return Some(p);
        }
    }
    let p = Path::new(r"C:\msys64\mingw64\bin").join(exe);
    p.exists().then_some(p)
}

fn main() {
    // Declare the cfg so `unexpected_cfgs` stays quiet on all toolchains
    // that understand the directive.
    println!("cargo:rustc-check-cfg=cfg(sf_cat_cxx)");
    println!("cargo:rerun-if-changed=csrc/transpose.cpp");
    println!("cargo:rerun-if-env-changed=SF_CAT_CXX");

    let dir = std::env::var("SF_CAT_CXX").ok();
    let (Some(cxx), Some(ar)) = (
        find_tool(dir.as_deref(), "g++.exe"),
        find_tool(dir.as_deref(), "ar.exe"),
    ) else {
        println!(
            "cargo:warning=sf-cat: MinGW g++/ar not found (set SF_CAT_CXX=<toolchain dir>); \
             C++ kernel lane disabled, Rust scalar path in use"
        );
        return;
    };

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let obj = out_dir.join("transpose.o");
    let lib = out_dir.join("libsf_cat_cxx.a");

    // Prepend the toolchain dir (parent of g++.exe) to PATH so cc1plus/as
    // can load the MinGW runtime DLLs.
    let mingw_bin = cxx
        .parent()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let path_env = format!(
        "{};{}",
        mingw_bin,
        std::env::var("PATH").unwrap_or_default()
    );

    let compiled = Command::new(&cxx)
        .args([
            "-std=c++17",
            "-O3",
            "-mavx2",
            "-fno-exceptions",
            "-fno-rtti",
            "-c",
        ])
        .arg("csrc/transpose.cpp")
        .arg("-o")
        .arg(&obj)
        .env("PATH", &path_env)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    let archived = compiled
        && Command::new(&ar)
            .arg("rcs")
            .arg(&lib)
            .arg(&obj)
            .env("PATH", &path_env)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

    if !archived {
        println!(
            "cargo:warning=sf-cat: C++ kernel compile/archive failed; Rust scalar path in use"
        );
        return;
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=sf_cat_cxx");
    println!("cargo:rustc-cfg=sf_cat_cxx");
}
