# CPU codec demonstration

`molecules.txt` contains ten manually selected molecular strings: acyclic,
branched, aromatic, cyclic and multiple-fragment-tree examples. This is a
software demonstration dataset, separate from the study's benchmark splits.
No downloaded dataset, pretrained model, GPU or network connection is used at runtime.

From the repository root, after installing `envs/requirements-demo.txt`:

```bash
python scripts/core/run_codec_demo.py
diff -u examples/codec/expected_roundtrips.tsv results/demo/roundtrips.tsv
```

Expected terminal output:

```text
SUMMARY	passed=10	total=10	failed=0
```

The tab-separated output contains input SMILES, encoded STAMP, recovered
canonical SMILES and an `OK` status for every molecule. The reference file was
generated with Python 3.12.12 and RDKit 2023.09.6. Exact STAMP strings depend on
RDKit canonical atom ordering; keep the demo version pins for file comparison.

The output file must not already exist. For subsequent runs, choose a new file:

```bash
python scripts/core/run_codec_demo.py --output results/demo/roundtrips_second.tsv
```

For your own data, provide a UTF-8 text file containing one SMILES per nonempty
line, without a header:

```bash
python scripts/core/run_codec_demo.py --input my_molecules.txt --output results/my_roundtrips.tsv
```

Exit status is 0 when all molecules roundtrip, 2 on a failed molecular
roundtrip, and nonzero on input/output errors. Existing outputs are protected
against replacement.
