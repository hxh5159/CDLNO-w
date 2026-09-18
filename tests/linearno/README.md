# LinearNO L1 tests

From repository root:

```bash
PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 python -B -m unittest \
  linearno.test_profiles linearno.test_schema linearno.test_legacy linearno.test_rng -v
```

These test independent schema/profile protocols and the **existing Transolver**.
No LinearNO operator, placeholder model, task entry import, production launcher,
real dataset, or external pickle is used. CPU tests use real PyG `Data`; original
hard-coded `.cuda()` position builders receive a test-process-only CPU shim.
Fixture state/inputs are seed-recomputed; outputs use atol=1e-6, rtol=1e-5, and
same-process strict checkpoint roundtrips require exact equality. Weight/key hashes
are strict; investigate framework initialization changes instead of overwriting
references when hashes differ.

`capture_fixtures.py /NEW/absolute/file.json` refuses overwrite. The committed small
text fixture includes complete output, config, weight/input hashes, parameter keys
and shapes, generator source hash and runtime version. No large model binaries are
committed. `rng_worker.py` continuation is test-only, using the existing RNG and
rendering primitives; it does not claim that old task entries have resume support.
Plasticity uses the actual Torch collate permutation, with NumPy tested separately.
See `docs/LINEARNO_L1_REPORT.md` for full results and boundaries.
