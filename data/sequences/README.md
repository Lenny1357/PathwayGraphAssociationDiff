# Sequences and structures for the pathway node space

Every node in the graphs this project generates is an entity with a real
molecular identity. This folder fetches it: a **CDS** for every gene, an
**amino-acid sequence** for every enzyme, and a **SMILES** string for every
metabolite, in both organisms.

```bash
python build_sequences.py --organism ecoli --what all
python build_sequences.py --organism human --what cds     # slowest step
```

`--what` takes any comma-separated subset of `cds`, `protein`, `ec`, `smiles`.
Everything lands in `seq_ref/`, with raw downloads cached in `seq_cache/` so a
rerun costs nothing.

## Coverage

| | genes / CDS | enzymes / AA | by EC code | metabolites / SMILES |
|---|---|---|---|---|
| **E. coli** (iML1515) | 1,513 / 1,516 (100%) | 1,514 / 1,516 (100%) | 651 / 917 EC → 3,020 proteins | 1,017 / 1,169 (87%) |
| **Human** (Human-GEM) | 2,844 / 2,848 (100%) | 2,847 / 2,848 (100%) | 913 / 1,139 EC → 6,731 proteins | 3,227 / 4,165 (77%) |

Sanity checks on the CDS: every E. coli sequence and 2,838 of 2,844 human ones
have a length divisible by three. 1,384 of 1,513 E. coli CDS begin ATG — the
rest begin GTG or TTG, which is ordinary for bacteria, not a fetch error.

The EC files resolve fewer codes than were queried because the query is
restricted to `reviewed:true`: an EC with no reviewed protein in that organism
returns nothing. Drop the restriction in `uniprot_by_ec` to trade precision
for coverage.

Compartments are collapsed for SMILES — iML1515's 1,877 metabolites are only
1,169 distinct chemical species, and `atp_c` and `atp_p` are the same molecule.

## Where each thing comes from, and why

| | source | why not the obvious alternative |
|---|---|---|
| gene CDS, human | Ensembl REST, canonical transcript then CDS | via the **batch** endpoints: 2,848 genes is 5,696 serial calls the naive way, about 65 batched |
| gene CDS, E. coli | NCBI ASM584v2 `cds_from_genomic.fna.gz`, 1.4 MB | E. coli K-12 is in Ensembl *Bacteria*, which `rest.ensembl.org` does not serve; one genome file also beats 1,516 REST calls |
| enzyme AA | UniProt by accession | both GEMs already carry a UniProt accession per gene (1,515/1,516 for iML1515), so this is exact |
| enzyme AA by EC | UniProt `ec:X AND organism_id:Y AND reviewed:true` | kept in a **separate** file: an EC names an activity, not a gene, so it returns every isozyme — many-to-many by nature |
| metabolite SMILES | MetaNetX → PubChem → ChEBI | see below |

## SMILES took four sources, not one

The naive route — look up the model's own `metanetx.chemical` annotation in
`chem_prop.tsv` — finds **42%** of iML1515. Each of the next three steps is
there because it fixed a measured gap:

1. **MetaNetX deprecation chain** (`chem_depr.tsv`, 2 MB). 646 of iML1515's
   1,114 annotated species point at MNXM ids that no longer exist. Following
   the chain: 42% → 73%.
2. **BiGG cross-reference** (`chem_xref.tsv`, 678 MB). Covers metabolites the
   model never annotated with an MNX id at all; recovered 3,040 of 3,044.
3. **PubChem**, by InChIKey and by CID. MetaNetX keeps *structure-less
   placeholder rows*: a BiGG compound it could not reconcile still gets an
   MNXM id, with empty InChI and SMILES columns, and the deprecation chain
   leads straight to it. `(R)`-propane-1,2-diol is one — `MNXM90191` →
   `MNXM1364597`, a row whose only content is `biggM:12ppd__R`. So "resolved
   to an MNX id" and "has a structure" are different questions. 73% → 86%.
4. **ChEBI** `structures.tsv.gz` (89 MB). Human-GEM annotates ChEBI far more
   often than PubChem (263 unresolved metabolites against 177), so this is
   what carries the human side. E. coli 86% → 87%, human 70% → 77%.

The `source` column in each CSV records which route answered.

Two implementation notes that are easy to get wrong. iML1515 annotates
InChIKeys and no CIDs; Human-GEM annotates CIDs and ChEBI ids and no
InChIKeys — both routes are needed to cover both organisms. And Human-GEM
packs several ids into one field as `MNXM150165;MNXM27815`, which resolves to
nothing if passed through whole.

What is left unmatched is mostly not a lookup failure: ACP-bound acyl chains,
Fe-S clusters and other `R`/`X` placeholder species have no single structure
to fetch — the same 97 metabolites `data/gem` excludes from its mass balance.

## The Ensembl path is resumable

Ensembl returns transient 500s and 503s often enough that a full 2,848-gene
run has a real chance of dying partway — the first attempt at this did, after
30 minutes of successful calls. So `_ensembl_batch`:

- **checkpoints** to `seq_cache/human_canonical.json` and `human_cds.json`
  after every batch, and a rerun starts from what is already cached;
- **adapts**: a batch that still fails after `_req`'s retries is split in half
  and each half retried, down to single ids. One dead id then costs one
  request instead of a whole chunk.

Rerunning the same command after a failure is always the right move.

## One gotcha worth knowing about

Human-GEM's `geneUniProtID` column is not always an accession — ENSG00000135070
carries the gene *symbol* `ISCA1` — and UniProt's `/accessions` endpoint
rejects the **whole batch of 100** over one malformed id. So ids are validated
against the accession pattern before the request, and anything that fails is
re-resolved through `gene_exact:<symbol> AND organism_id:<taxid>`.

## The single-id API

`get_transcripts`, `get_canonical_transcript` and `get_cds` are kept as
one-gene-at-a-time functions for checking something by hand. The bulk path
does **not** call them — it uses the POST endpoints, roughly 90x fewer
requests for the same answer — but they are the readable reference for what
that path does.

## Files

| file | contents |
|---|---|
| `{org}_gene_cds.fasta` | CDS, keyed by the project's gene node id (`b0002`, `ENSG…`) |
| `{org}_enzyme_aa.fasta` | protein, keyed by the same gene id |
| `{org}_ec_enzyme_aa.fasta` | protein, keyed `EC|accession` |
| `{org}_ec_index.json` | `{EC: [EC\|accession, …]}` |
| `{org}_metabolite_smiles.csv` | `node_id,name,mnx_id,formula,smiles,inchikey,source` |
| `manifest.json` | per-step counts for everything above |
