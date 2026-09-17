"""Guard paper-transcribed documentation against regression to older results.

Reference: the anonymous manuscript revision supplied on 2026-09-17.
These checks validate transcription, not experimental reproduction.
"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ReadmePaperRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = (ROOT / "README.md").read_text()

    def test_method_name_and_main_result_match_revision(self):
        self.assertIn("# CoER\n", self.readme)
        self.assertIn("Co-Evolution and Refinement", self.readme)
        self.assertIn(
            "| **CoER** | **79.62** | **79.76** | **0.00** | **79.76** | **75.40** | **0.25** | **75.40** |",
            self.readme,
        )
        self.assertIn("3 compromised adaptive executions out of 1,187", self.readme)
        self.assertIn("38.45% to 0.22%", self.readme)

    def test_agentlab_joint_metrics_match_paper_counts(self):
        counts = {
            "Base": (355, 564, 359),
            "Co-PPO": (150, 746, 668),
            "CoER": (134, 786, 732),
        }
        for name, numerators in counts.items():
            values = " | ".join(f"{100 * n / 949:.2f}" for n in numerators)
            self.assertIn(f"| {name} | {values} |", self.readme)

    def test_injecagent_preserves_denominators_and_missing_counts(self):
        self.assertIn("| Base | 7.49 | 77/1,028 | 22.69 | 221/974 |", self.readme)
        self.assertIn("| Co-PPO | 4.40 | Not supplied | 17.30 | Not supplied |", self.readme)
        self.assertIn("| CoER | 0.00 | 0/1,043 | 1.97 | 20/1,016 |", self.readme)
        self.assertIn("scope of excluded cases remains unverified", self.readme)

    def test_resources_are_not_misrepresented_as_uploaded(self):
        self.assertIn("No public download is claimed yet", self.readme)
        self.assertIn("3,995 conversations; 11,655 supervised attacker turns", self.readme)
        self.assertIn("5,760 trajectories: 4,907 attacked + 853 untriggered replay", self.readme)
        self.assertIn("12,705 training rows + 3,186 disjoint internal-validation rows", self.readme)
        self.assertIn("training and validation as separate splits", self.readme)

    def test_only_two_original_figures_and_current_provenance(self):
        images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", self.readme)
        self.assertEqual(
            images,
            ["docs/assets/paper/figure2-corl-framework.png", "docs/assets/paper/figure1-adaptive-ipi.png"],
        )
        provenance = (ROOT / "docs/assets/paper/README.md").read_text()
        self.assertIn("2026-09-17", provenance)
        self.assertIn("ce2ef1e78e8e1d0d30ee48ce2850d1d205bbf560ddfb0be95ae55de8f8428439", provenance)

    def test_release_gitignore_preserves_model_implementation_source(self):
        from scripts.export_anonymous import collect_files

        self.assertIn(Path(".gitignore"), collect_files(ROOT))
        patterns = (ROOT / ".gitignore").read_text().splitlines()
        self.assertIn("/models/", patterns)
        self.assertNotIn("models/", patterns)
        self.assertIn("*.safetensors", patterns)


if __name__ == "__main__":
    unittest.main()
