# Data sources and origins

Every external dataset this project consumes, where it comes from, what it is
pinned to, and which script turns it into the tensors the models read.

Each `data/<source>/` directory has its own README with the per-script detail; this
file is the index across them, and the place that records **what is and is not
reproducible from this repository**.

---

## At a glance

| Source | Organism | What it provides | Built by | Local artefact |
| --- | --- | --- | --- | --- |
| [Human-GEM](https://github.com/SysBioChalmers/Human-GEM) | human | genome-scale metabolic model — stoichiometry, subsystems, gene-reaction rules | `data/gem/build_gem.py` | `data/gem/gem_ref/` (176 MB) |
| [BiGG iML1515](http://bigg.ucsd.edu/models/iML1515) | *E. coli* K-12 MG1655 | genome-scale metabolic model | `data/gem/build_gem_ecoli.py` | `data/gem/gem_ref_ecoli/` (12 MB) |
| [BiGG iML1515](http://bigg.ucsd.edu/models/iML1515) | *E. coli* | reference network + pathway sampler | `data/ecoli/build_reference.py`, `build_pathways.py` | `data/ecoli/ecoli_ref/` (60 MB) |
| [PRECISE-1K](https://github.com/SBRG/precise1k) | *E. coli* | 1,035-sample expression compendium → co-expression | `data/ecoli/build_coexpression.py` | `data/ecoli/ecoli_expr_nodespace/` (108 MB) |
| [Replogle 2022 Perturb-seq](https://doi.org/10.25452/figshare.plus.20029387) | human (K562, RPE1) | CRISPRi knockdown → transcriptome effect matrix | `data/perturbseq/build_perturbseq.py` | `data/perturbseq/perturb_ref/` (44 MB) |
| [EnzymeMap](https://github.com/hesther/enzymemap) | multi | 349k curated enzymatic reactions, EC codes, SMILES | `data/enzymemap/build_enzymemap.py` | `data/enzymemap/` (96 MB) |
| Ensembl REST, UniProt, PubChem, ChEBI, NCBI | human, *E. coli* | gene CDS, protein sequences, metabolite SMILES | `data/sequences/build_sequences.py` | `data/sequences/` (1.6 GB) |
| [KEGG REST](https://rest.kegg.jp) | multi | curated modules as ready-made pathways | `data/kegg/build_kegg.py` | (optional alternative dataset) |
| SRA-derived tissue co-expression | human | per-tissue gene co-expression + expression | **not built here** — see below | `$PT_FILES/*.pt` |

---

## The two training datasets

Everything the models train on is a set of 64-node subgraphs cut from a
genome-scale metabolic network. Nodes are metabolites, reactions, genes and
enzymes; edges are participation relations from the model's stoichiometry and
gene-reaction rules.

**Human** — Human-GEM, pulled from the `main` branch of the SysBioChalmers
repository (`Human-GEM.mat`, `reactions.tsv`, `metabolites.tsv`, `genes.tsv`).
Note that this is a **branch, not a tagged release**, so a rebuild at a later
date may not reproduce the current vocabulary.

**E. coli** — iML1515 from BiGG (`iML1515.json`), chosen over the `.mat`
because it is the canonical distribution. `build_gem_ecoli.py` verifies itself
by solving the growth LP and comparing against the published optimum
(µ = 0.8770 h⁻¹), and mass-balances 2,200/2,200 checkable reactions.

Identifiers key on **b-numbers** (`b4025`) throughout the E. coli pipeline —
iML1515's genes, PRECISE-1K's rows and RegulonDB all use them, so there is no
ID-mapping step and therefore no join loss.

---

## Conditioning and evaluation signals

These are not trained on. They are used to *score* generated pathways, and in
`conditioning_test.ipynb` to condition generation.

### Co-expression

**E. coli — PRECISE-1K.** 1,035 RNA-seq samples with curated metadata, from
`github.com/SBRG/precise1k` (`log_tpm_qc.csv`, `metadata_qc.csv`). E. coli has
no tissues, so the axis the human pipeline calls "tissue" becomes **condition
strata** taken from PRECISE-1K's own metadata — stratified by `project`,
giving 24 networks over 1,410 genes, top-32 partners each at |r| ≥ 0.3.

**Human — SRA-derived tissue co-expression.** 11 tissues (lung, prostate,
skin, small intestine, breast, bone marrow, testis, spleen, pancreas, ovary,
adipose). Consumed by `graphpath/data/expression.py` as
`co_expression_sparse_homo_sapiens_sra_<tissue>.pt` plus
`tissue_expression_dict.pkl`.

### Perturbation response

**Replogle et al. 2022**, genome-scale Perturb-seq, figshare 20029387. The
**pseudobulk** matrices are used rather than the raw single-cell data — 375 MB
against 66 GB, and the pseudobulk is what a per-gene effect profile needs.

| figshare file | dataset | scale |
| --- | --- | --- |
| 35773217 | `gwps` — genome-wide K562 | 9,852 perturbations |
| 35780870 | `essential` — K562 essential genes | 2,057 perturbations |
| 35775512 | `rpe1` — RPE1 essential genes | a second cell line |

Joined to Human-GEM this yields an **847 × 8,175** effect matrix over
metabolic genes, after dropping knockdowns with no significant transcriptional
phenotype (`energy_test_p_value`), which roughly halves the set — an
unaffected knockdown's profile is measurement noise and would sit isolated.

Two derived networks, in **different index spaces** — mixing them silently
mislabels every node:

- `similarity.npz` — rows correlated. An edge means *two genes whose knockdowns
  break the same thing*. Nodes index `meta["genes"]` (847 perturbed).
- `similarity_response.npz` — columns correlated. An edge means *two genes that
  react alike across the panel*. Nodes index `meta["response_genes"]`
  (~1,445 measured Human-GEM genes). Self-knockdown cells are masked first;
  they are ~9× a typical entry and would otherwise dominate a column of 847.

### Sequences and structures

`data/sequences/build_sequences.py` assembles, for both organisms:

| Output | Source |
| --- | --- |
| gene CDS (FASTA) | Ensembl REST (human), NCBI RefSeq `GCF_000005845` (*E. coli*) |
| protein sequences | UniProt REST, resolved by EC number and organism |
| metabolite SMILES | PubChem PUG REST, ChEBI flat files, MetaNetX cross-references |

### Curated reaction chemistry

**EnzymeMap** (`github.com/hesther/enzymemap`, `main` branch) — 349,458 atom-
mapped enzymatic reactions, 4,552 EC codes, 14,042 UniProt sequences. Used for
EC-level annotation and molecule structures. InChIKey matching is done on the
**skeleton block** (the part before the first hyphen) so protonation states do
not split otherwise-identical molecules.

---

## Reproducibility: what this repo can and cannot rebuild

**Fully reproducible.** Everything under `data/ecoli/`, `data/gem/`,
`data/perturbseq/`, `data/enzymemap/`, `data/sequences/` and `data/kegg/`
downloads its own inputs and rebuilds its artefacts from a documented URL.

**Not reproducible from this repository.** The human training inputs and the
human co-expression tables are *consumed* but never *built* here. They are
expected as pre-existing files under `$PT_FILES`
(default `pt_files`):

| File | Consumed by |
| --- | --- |
| `human_gem_sampled_graphs.pkl` | the training datasets in every notebook |
| `node_id_map.pkl` | the shared vocabulary |
| `node_name_map.csv` | common-name lookup |
| `human_gem_reaction_node_edges2.csv` | `ReferenceNetwork` |
| `co_expression_sparse_homo_sapiens_sra_<tissue>.pt` | `ExpressionLoader` |
| `tissue_expression_dict.pkl` | `ExpressionLoader` |

The E. coli equivalents *do* have builders (`data/ecoli/build_reference.py`,
`build_pathways.py`, `build_coexpression.py`), and `data/kegg/build_kegg.py`
builds a third dataset end to end including the pathway sampler. The human
sampler and co-expression builder are the gap — if those files are lost, the
human half of the project cannot currently be regenerated.

---

## Per-directory documentation

| Directory | README |
| --- | --- |
| E. coli dataset | [data/ecoli/README.md](data/ecoli/README.md) |
| FBA / stoichiometry | [data/gem/README.md](data/gem/README.md) |
| Perturb-seq | [data/perturbseq/README.md](data/perturbseq/README.md) |
| Sequences | [data/sequences/README.md](data/sequences/README.md) |
| EnzymeMap | [data/enzymemap/README.md](data/enzymemap/README.md) |
| KEGG alternative | [data/kegg/README.md](data/kegg/README.md) |
| Benchmarks | [graphpath/benchmarks/README.md](graphpath/benchmarks/README.md) |
| Architecture bench | [experiments/arch_bench/README.md](experiments/arch_bench/README.md) |
| Splits | [experiments/splits/README.md](experiments/splits/README.md) |
| Model architecture | [README.md](README.md) |
