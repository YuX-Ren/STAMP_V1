# STAMP software functionality

The codec converts molecular graphs to a motif-tree sequence and reconstructs
graphs from motif, joint and branch tokens. Vocabulary tools build reusable
motif inventories. Benchmark modules train and evaluate models using these
representations.

## Encoding and reconstruction

```text
encode(smiles):
    parse and sanitize the molecular graph with RDKit
    fragment the graph according to the selected codec configuration
    construct fragment adjacency and identify attachment atoms
    choose a root and traverse the fragment tree
    emit motif identities, attachment indices and branch spans
    return the serialized STAMP sequence

decode(tokens):
    parse motif, joint and branch groups
    place the root motif and initialize attachment and branch stacks
    attach subsequent motifs at the specified ports
    save and restore branch context when processing branch spans
    apply configured reconstruction fallbacks and sanitize the result
    return the reconstructed molecular graph or a decoding failure

roundtrip(smiles):
    tokens = encode(smiles)
    recovered = decode(tokens)
    compare canonical input and recovered molecular SMILES
```

Implementation: [`stamp/core/stamp_codec.py`](../stamp/core/stamp_codec.py),
[`stamp/core/reconstruction.py`](../stamp/core/reconstruction.py).
The full grammar and fallback semantics are described in
[the representation specification](stamp.md).

## Vocabulary construction

The builder reads molecular inputs, generates fragment candidates, aggregates
motif and attachment-schema statistics, scores candidates and exports the
selected vocabulary and metadata. Implementation:
[`stamp/vocab/vocab_builder.py`](../stamp/vocab/vocab_builder.py);
command: `python scripts/build_vocab.py --help`.

## Model workflows

The benchmark layer applies a representation to prepared molecular partitions,
fits or loads its vocabulary and model, trains the selected objective, and
evaluates generated or predicted outputs using task-specific metrics.
[`stamp/benchmark/lm.py`](../stamp/benchmark/lm.py) contains several distinct
model objectives; AO-GPT and masked diffusion are separate configurations.
[`MODEL_CARD.md`](../MODEL_CARD.md) maps model families to intended uses,
training sources, evaluation protocols and limitations.

This file documents repository functionality. The submitted manuscript's
Methods and supplementary algorithms must independently identify the matching
implementation and pseudocode locations for the editorial software form.
