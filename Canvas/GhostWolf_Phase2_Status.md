### Phase 2 — Voxel Engine Core

[Start of Phase 2]
- Scaffold created: CMake project, `engine` lib, `apps/cli`.
- Core modules added: `gw_engine`, `voxel_grid`, `rays` (4D 120-way split), `traversal`, `aggregation`, `scheduler`.
- Minimal C ABI and C++ facade implemented.

[Debug & Test]
- Fixed PIMPL member in `gw::Engine` and resolved ray type warning.
- Traversal emits group-based features to validate mean collapse (past=0.3, present=0.6, future=0.9).
- Build succeeds; CLI runs and shows stable output.

Example CLI run:
```bash
$ ./build/apps/cli/ghostwolf_cli
Step 0: token=60, lp=0.600001
Step 1: token=60, lp=0.600001
```

[End of Phase 2 — after debug & test]
- Phase 2 complete and verified; ready for Phase 3 (Python orchestration & bindings).