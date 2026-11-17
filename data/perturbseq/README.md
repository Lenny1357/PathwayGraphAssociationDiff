# Perturb-seq for Human-GEM

A **causal** functional network over metabolic genes, to replace the
co-expression conditioning that the E. coli analysis showed barely
distinguishes pathway members from random pairs.

| file | |
|---|---|
| `build_perturbseq.py` | download the Replogle pseudobulk → effect matrix + similarity network |
| `perturb_ref/` | the output: `effect_matrix.npy`, `similarity.npz`, `meta.json` |
| `pseudobulk/` | download cache |
| `data/perturbseq.ipynb` | exploratory plots and the Human-GEM integration |

```bash
python build_perturbseq.py --dataset gwps          # 375 MB, ~2 min
```

## Why this dataset

Co-expression answers "do A and B move together". Perturb-seq answers **"does
knocking down A change B"** — a causal, directed readout. Replogle et al. 2022
(figshare 20029387) knocked down 9,852 genes in K562 with a transcriptome-wide
readout, and published **pseudobulk** matrices alongside the raw data. The
pseudobulk is two orders of magnitude smaller — 375 MB against 66 GB — and is
all this needs.

Ids are ENSG on both sides, so the join to Human-GEM needs no mapping.

## Coverage and quality

```
2,848 Human-GEM genes
  1,626 perturbed (57.1%)     1,445 measured (50.7%)
    847 with a significant transcriptional phenotype   <- the usable set
```

Two genes whose knockdowns produce the same transcriptional consequence are
functionally related. Scoring that against shared Human-GEM subsystem, versus
the co-expression result from `ecoli_coexpression_information.ipynb`:

| | Cohen's d | AUC | @0.5% | @1% | @5% |
|---|---|---|---|---|---|
| **Perturb-seq profile similarity** | **0.456** | 0.574 | **5.7x** | **4.6x** | **2.9x** |
| E. coli co-expression | 0.131 | 0.579 | 3.1x | 2.5x | 1.8x |

**3.5x the effect size**, and unlike co-expression it holds up away from the
sparse tail — 2.9x at 5% coverage against 1.8x. The two are not strictly
comparable (different organism, different baseline rate), but the gap is far
larger than that caveat.

## Three filters that are not optional

**Keep only knockdowns that did something.** Replogle score every perturbation
for whether its transcriptome differs from control (`energy_test_p_value`).
Roughly half do not, and an unaffected knockdown's profile is measurement
noise: before filtering, 1,291 of 1,626 metabolic genes sat isolated in the
network; after, 64 of 847.

**Drop the readout genes with undefined values.** 6,881 entries are `+inf` —
measured genes whose control expression is ~0, so the ratio is undefined. They
sit in only 73 of 8,248 columns, so dropping those columns is cleaner than
patching the values.

**Do not reach for `np.nan_to_num`.** Its default maps `inf` to the dtype
maximum — 3.4e38 in float32 — which then overflows the moment anything squares
it. That silently corrupted the first similarity matrix built here: the
measured effect size read 0.287 instead of 0.456.

## Using it

`similarity.npz` is `row`/`col`/`val` indexing into `meta.json["genes"]`, the
same shape as the co-expression tables, so it drops into the conditioning path
without touching the model. `effect_matrix.npy` is the full
(847, 8,175) causal matrix if you want the directed version rather than the
symmetric similarity.

Two uses, and the second is the one co-expression cannot support:

- **symmetric**: `similarity.npz` as a functional gene-gene network
- **directed**: row *i* of the effect matrix is what knocking down gene *i*
  does to everything else — a causal edge set, not a correlational one

## The notebook

`data/perturbseq.ipynb`, six cells, runs from the built `perturb_ref/`:

1. what survives the phenotype and finiteness filters
2. the Human-GEM join, including **how uneven coverage is by subsystem**
3. does profile similarity recover subsystem membership (the figure above)
4. the functional network drawn, Louvain communities against Human-GEM subsystems
5. **directed** effects — the thing co-expression cannot represent

Two scale traps it documents, both of which produced wrong figures first time:

**The effects are small.** Median |value| 0.06, 99th percentile 0.43, max ~5.
A `|z| > 2` cut selects 26 of 571,536 entries in the metabolic block and
reports the threshold rather than the biology. The notebook uses magnitude
(total |effect|) and the matrix's own 99.9th percentile instead of a fixed cut.

**The sanity check that caught it:** ranked by total influence on other
metabolic genes, the top knockdowns come out as aminoacyl-tRNA biosynthesis,
nucleotide/purine/pyrimidine metabolism and the TCA cycle — central essential
processes whose loss reshapes the transcriptome. With the bad threshold the
"most influential" gene moved one other gene, which was the signal something
was wrong.

## Caveats

**One cell line.** K562 is an erythroleukemia line; its metabolism is not a
generic human cell's. `--dataset rpe1` gives a second line (RPE1, epithelial)
to check whether a result transfers, which is worth doing before believing
anything cell-type specific.

**Transcriptional readout only.** A metabolic knockdown often changes flux
without changing transcription, so a gene with no phenotype here is not
necessarily inert — it may simply be invisible to this assay. That is the
likely reason 779 of 1,626 perturbed metabolic genes were filtered out.

**Essential-gene bias in the alternatives.** `--dataset essential` covers only
2,057 perturbations and reaches just 11% of Human-GEM genes, since most
metabolic genes are not essential in K562. Use `gwps` for this purpose.
