#!/usr/bin/env bash
# Post-process the AppImage produced by `tauri build --bundles appimage`.
#
# Tauri runs linuxdeploy over the whole AppDir (sidecars included) with fixed arguments, which
#   - copies every host-resolved dependency into usr/lib, including the NVIDIA *driver* lib
#     libcuda.so.1 — it must come from the host (version-locked to the kernel module);
#   - rewrites the helpers' RUNPATH to $ORIGIN/../lib, so the CUDA runtime libs are loaded from
#     usr/lib and the Tauri resources copy in usr/lib/GhostReel/lib is a ~575 MB duplicate.
# This script removes driver libs, folds usr/lib/GhostReel/lib into usr/lib and repacks the
# AppImage with the appimage plugin Tauri already downloaded.
set -euo pipefail
cd "$(dirname "$0")/.."

target_dir="${CARGO_TARGET_DIR:-target}"
out_dir="$target_dir/release/bundle/appimage"
appdir="$(find "$out_dir" -maxdepth 1 -name '*.AppDir' -type d | head -n1)"
appimage="$(find "$out_dir" -maxdepth 1 -name '*.AppImage' -type f | head -n1)"
[ -n "$appdir" ] && [ -n "$appimage" ] || { echo "no AppDir/AppImage in $out_dir (run tauri build --bundles appimage)" >&2; exit 1; }
plugin="${TAURI_TOOLS_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/tauri}/linuxdeploy-plugin-appimage.AppImage"
[ -x "$plugin" ] || { echo "missing $plugin (Tauri downloads it on the first AppImage build)" >&2; exit 1; }

lib="$appdir/usr/lib"
# 1. Driver libs always come from the host.
rm -fv "$lib"/libcuda.so* "$lib"/libnvidia-*.so*

# 2. One copy of the CUDA runtime libs, where the rewritten RUNPATH ($ORIGIN/../lib) looks.
res_lib="$lib/GhostReel/lib"
if [ -d "$res_lib" ]; then
  for f in "$res_lib"/*; do
    [ -e "$f" ] || continue
    name="$(basename "$f")"
    [ -e "$lib/$name" ] || mv "$f" "$lib/$name"
  done
  rm -rf "$res_lib"
fi

# 3. Sanity: helpers must still resolve their CUDA libs inside the AppDir.
for h in "$appdir"/usr/bin/ghostreel-asr "$appdir"/usr/bin/ghostreel-llm; do
  [ -f "$h" ] || continue
  rp="$(readelf -d "$h" | sed -n 's/.*R\(UN\)\?PATH.*\[\(.*\)\]/\2/p')"
  case "$rp" in
    *'$ORIGIN/../lib'*) ;;
    *) echo "warning: $h RUNPATH is '$rp' (expected \$ORIGIN/../lib)" >&2 ;;
  esac
done

# 4. Repack.
rm -f "$appimage"
ARCH=x86_64 OUTPUT="$appimage" APPIMAGE_EXTRACT_AND_RUN=1 "$plugin" --appdir "$appdir"
echo "repacked $appimage ($(du -h "$appimage" | cut -f1))"
