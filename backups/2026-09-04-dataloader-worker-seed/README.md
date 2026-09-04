# DataLoader worker RNG fix: pre-fix backup

The production source snapshot predates the worker RNG fix and comes from Git
commit `be82ed062bece794b177505415714118b0277540`.

## Backup contents

- `production-sources.before.tar`: exact pre-fix copies of
  `datasets/dehaze.py` and `training/runtime.py`.
- `dehaze.py.before`: directly readable pre-fix copy of `datasets/dehaze.py`.
- `test_unpaired_dataset.py.before`: pre-fix copy of the dataset tests.

## SHA-256

- `production-sources.before.tar`:
  `2a022998b6bce9e72d5170e14306fcebf2ad919dcb8e2048ae8a7a79843690b3`
- original `datasets/dehaze.py`:
  `4fc3eae48c96eba273e70979fc96cde485712b4ead9b2964cb5738d96f59ecdd`
- original `training/runtime.py`:
  `77b1225cf0e382cba0d2d36ff48951fbc80028249780e2daffe4faef527dac54`
- original `tests/test_unpaired_dataset.py`:
  `f6be050fcd2fddd9b7730019f405db68dbb14e1528a4190e82db7b7f8ef72415`

The archive can be inspected without touching the working tree with:

```bash
tar -tf backups/2026-09-04-dataloader-worker-seed/production-sources.before.tar
```
