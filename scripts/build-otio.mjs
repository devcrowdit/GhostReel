#!/usr/bin/env node
// Build the ghostreel-otio sidecar (PyInstaller one-file) into target/sidecars/.
//
// The export path (FCP7 XML / .otio) runs through OpenTimelineIO, which is Python; the bundle
// carries a frozen copy so a released app needs no Python. tools/ghostreel-otio/build.sh does the
// same thing on Linux only — this one also runs on the Windows runner, where bash and `python3`
// are not a given. Uses node: builtins only.

import { existsSync, mkdirSync, rmSync, readFileSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { createHash } from 'node:crypto';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const toolDir = join(repoRoot, 'tools', 'ghostreel-otio');
const targetDir = process.env.CARGO_TARGET_DIR ?? join(repoRoot, 'target');
const distDir = join(targetDir, 'sidecars');
const isWindows = process.platform === 'win32';
const exeExt = isWindows ? '.exe' : '';
const outPath = join(distDir, `ghostreel-otio${exeExt}`);
const stampPath = join(distDir, 'ghostreel-otio.stamp');
const force = process.argv.includes('--force');

// The frozen binary only changes when the script, its requirements or the Python major version do.
const stamp = createHash('sha256')
  .update(readFileSync(join(toolDir, 'ghostreel_otio.py')))
  .update(readFileSync(join(toolDir, 'requirements.txt')))
  .update(readFileSync(join(toolDir, 'requirements-build.txt')))
  .digest('hex')
  .slice(0, 16);

if (!force && existsSync(outPath) && existsSync(stampPath) && readFileSync(stampPath, 'utf8').trim() === stamp) {
  console.log(`ghostreel-otio is up to date (${outPath})`);
  process.exit(0);
}

function run(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { stdio: 'inherit', cwd: repoRoot, ...opts });
  if (r.error) throw r.error;
  if (r.status !== 0) {
    console.error(`Error: ${cmd} ${args.join(' ')} exited with ${r.status}`);
    process.exit(1);
  }
}

// Find a usable interpreter: `python3` everywhere but Windows, where it is a Store stub.
function findPython() {
  const candidates = process.env.PYTHON ? [process.env.PYTHON] : isWindows ? ['python', 'py'] : ['python3', 'python'];
  for (const c of candidates) {
    const args = c === 'py' ? ['-3', '--version'] : ['--version'];
    const r = spawnSync(c, args, { stdio: 'pipe', encoding: 'utf8' });
    if (r.status === 0) return { cmd: c, prefix: c === 'py' ? ['-3'] : [] };
  }
  console.error('Error: no Python 3 interpreter found (set PYTHON to one).');
  console.error('The otio sidecar is what exports a script to Premiere/Resolve; without it the');
  console.error('bundle ships with Export disabled.');
  process.exit(1);
}

const py = findPython();
const venvDir = join(toolDir, '.build-venv');
const venvBin = join(venvDir, isWindows ? 'Scripts' : 'bin');
const venvPy = join(venvBin, `python${exeExt}`);

if (!existsSync(venvPy)) {
  rmSync(venvDir, { recursive: true, force: true });
  console.log('Creating build venv...');
  run(py.cmd, [...py.prefix, '-m', 'venv', venvDir]);
}

console.log('Installing OpenTimelineIO and PyInstaller...');
run(venvPy, ['-m', 'pip', 'install', '--quiet', '--upgrade', 'pip']);
run(venvPy, [
  '-m', 'pip', 'install', '--quiet',
  '-r', join(toolDir, 'requirements.txt'),
  '-r', join(toolDir, 'requirements-build.txt'),
]);

mkdirSync(distDir, { recursive: true });
console.log('Freezing ghostreel-otio...');
run(venvPy, [
  '-m', 'PyInstaller',
  '--onefile',
  '--noconfirm',
  '--name', 'ghostreel-otio',
  '--collect-all', 'opentimelineio',
  '--collect-all', 'otio_fcp_adapter',
  '--copy-metadata', 'otio-fcp-adapter',
  '--distpath', distDir,
  '--workpath', join(targetDir, 'sidecars-build'),
  '--specpath', join(targetDir, 'sidecars-build'),
  join(toolDir, 'ghostreel_otio.py'),
]);

if (!existsSync(outPath)) {
  console.error(`Error: PyInstaller finished but ${outPath} is missing.`);
  process.exit(1);
}

// A binary that cannot answer is worse than none: check it runs before anyone bundles it.
const probe = spawnSync(outPath, ['validate', '--help'], { stdio: 'pipe', encoding: 'utf8' });
if (probe.status !== 0) {
  console.error(`Error: ${outPath} does not run (exit ${probe.status}): ${probe.stderr ?? probe.error}`);
  process.exit(1);
}

writeFileSync(stampPath, `${stamp}\n`);
console.log(`Built ${outPath}`);
