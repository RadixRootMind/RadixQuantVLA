from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_awesome_quant_vla.sh"


class FoldQuantIntegrationTest(unittest.TestCase):
    def test_foldquant_profiles_resolve(self) -> None:
        cases = [
            ("pi05_foldquant_w4a4", "pi05", "w4a4_srg + w4a4_shg"),
            ("pi05_foldquant_w8a8", "pi05", "w8a8_sr + w8a8_sh"),
            ("groot_n15_foldquant_w4a4", "groot_n1_5", "w4a4_srg + w4a4_shg"),
            ("groot_n15_foldquant_w8a8", "groot_n1_5", "w8a8_sr + w8a8_sh"),
        ]
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is not available")
        for profile, family, schemes in cases:
            with self.subTest(profile=profile):
                completed = subprocess.run(
                    [bash, str(LAUNCHER), profile, "--suite", "spatial", "--action", "plan"],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn(f"foldquant family : {family}", completed.stdout)
                self.assertIn(f"schemes          : {schemes}", completed.stdout)

    def test_foldquant_is_external_model_integration(self) -> None:
        pi_upstream = (
            ROOT
            / "third_party"
            / "foldquantvla"
            / "adapters"
            / "pi05"
            / "foldquant_integration"
            / "_upstream.py"
        ).read_text(encoding="utf-8")
        groot_upstream = (
            ROOT
            / "third_party"
            / "foldquantvla"
            / "adapters"
            / "groot_n1_5"
            / "foldquant_integration"
            / "_upstream.py"
        ).read_text(encoding="utf-8")
        self.assertIn("FOLDQUANT_OPENPI_ROOT", pi_upstream)
        self.assertIn("OPENPI_ROOT", pi_upstream)
        self.assertIn("FOLDQUANT_LIBERO_DIR", pi_upstream)
        self.assertIn("FOLDQUANT_GROOT_ROOT", groot_upstream)
        self.assertIn("AWESOME_QVLA_ROOT", groot_upstream)

    def test_foldquant_provenance_and_licenses_are_present(self) -> None:
        root = ROOT / "third_party" / "foldquantvla"
        self.assertTrue((root / "LICENSE").is_file())
        self.assertTrue((root / "NOTICE").is_file())
        provenance = (root / "UPSTREAM.md").read_text(encoding="utf-8")
        self.assertIn("7d98edd705c33ef994b3caf8c2c93d28da78413a", provenance)
        self.assertIn("cair-vinuni/FoldQuantVLA", provenance)

    def test_foldquant_cuda_target_include_compatibility_is_present(self) -> None:
        build = (
            ROOT / "third_party" / "foldquantvla" / "foldquant" / "kernels" / "build.py"
        ).read_text(encoding="utf-8")
        self.assertIn("CUDAToolkit_INCLUDE_DIR={include}", build)
        self.assertIn('root / "targets" / f"{platform.machine()}-linux" / "include"', build)

    def test_launchers_use_lf_line_endings(self) -> None:
        for path in (LAUNCHER, ROOT / "scripts" / "run_foldquantvla.sh"):
            self.assertNotIn(b"\r\n", path.read_bytes(), f"CRLF line endings in {path}")


if __name__ == "__main__":
    unittest.main()
