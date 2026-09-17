// Bundled CUDA runtime libs (cudart, cuBLAS, cuBLASLt) are found via RUNPATH, never via the
// host toolkit or LD_LIBRARY_PATH (plan §8, `scripts/stage-helpers.mjs`):
//   $ORIGIN/lib                  CLI tarball: helper next to lib/
//   $ORIGIN/../lib/GhostReel/lib deb/AppImage: helper in usr/bin, Tauri resources in usr/lib/GhostReel
// Set here rather than in .cargo/config.toml rustflags: it only relinks this crate, while a
// global rustflags change would rebuild every crate (incl. the CUDA ggml builds).
// Missing dirs are harmless, so dev builds keep resolving the system toolkit.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN/lib:$ORIGIN/../lib/GhostReel/lib");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
