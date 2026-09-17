// Bundled CUDA runtime libs are found via RUNPATH — see crates/ghostreel-asr/build.rs.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN/lib:$ORIGIN/../lib/GhostReel/lib");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
