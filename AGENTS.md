# Repository instructions

- `PROJECT_EVOLUTION_RECORD.md` is the canonical cross-stage implementation and research history.
- Any change that adds or modifies functionality, contracts, experiments, evaluation results, known defects, fixes, conclusions, or next-stage plans must update `PROJECT_EVOLUTION_RECORD.md` in the same change.
- After every update, copy the complete canonical record to `../../local-history/PROJECT_EVOLUTION_RECORD.md` (relative to this repository root) and verify that both files have the same SHA-256. Never update only one copy. This local mirror replaces the previous computer's `C:\Users\12204\...` location.
- `PROJECT_EVOLUTION_RECORD.md` is a local-only research log. Keep it excluded through `.git/info/exclude`; do not add it to Git or GitHub.
- Keep stage-specific technical documents for detail, but keep the root record synchronized with the current project state.
- Distinguish deterministic engineering proof, directional real-model evidence, and statistically supported research conclusions.
