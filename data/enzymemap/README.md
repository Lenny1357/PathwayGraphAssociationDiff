# EnzymeMap: what reactions *do*, not just that they exist

EnzymeMap (Heid, Probst, Green & Madsen, *Chem. Sci.* 2023) is BRENDA's
enzymatic reactions curated, atom-mapped and validated. This folder downloads
it, fetches a protein sequence for every UniProt accession it cites, and
bridges it onto the two GEMs this project generates pathways over.

```bash
python build_enzymemap.py                      # everything, ~15 min
python build_enzymemap.py --what parse,bridge  # skip the UniProt fetch
```

## Why it is here

The GEMs say a reaction **exists**. EnzymeMap says what it **does** — which
bonds move, and which protein moves them.

That matters for the specific hole in the current benchmarks. `edge_precision`
in `compare_models_ecoli.ipynb` scores a generated edge against the whole
iML1515 edge table, 98.5% of which the model saw in training, so it measures
recall of training data rather than chemical plausibility. EnzymeMap can
judge a proposed enzyme–reaction or metabolite–reaction edge on whether the
transformation is one that EC class is *known to catalyse*, in any organism —
a reference that is independent of this project's training split.

## What you get

| file | contents |
|---|---|
| `reactions.csv.gz` | 349,458 reactions × 11 columns |
| `proteins.fasta` | 14,042 UniProt sequences, keyed by accession |
| `ec_index.csv` | per EC: reactions, organisms, proteins, and whether each GEM uses it |
| `species.csv` | 42,940 distinct molecules: InChIKey, canonical SMILES, use count, GEM node id |
| `manifest.json` | counts for every step |

The `rule` and `mapped` columns are dropped from the derived table — they are
most of the 386 MB uncompressed and neither is needed to ask which EC, protein
or molecule is involved. `em_cache/processed_reactions.csv.gz` keeps them.

## The dataset

```
349,458 reactions   214,830 distinct rxn_idx   4,552 EC codes   7,468 organisms
quality             median 1.000, 86% >= 0.9
natural             77,938 natural / 271,520 not
steps               334,529 single, 9,799 single-from-multi, 5,130 multi
source              244,475 direct, the rest reversed or rule-suggested
protein_db          80,429 uniprot, 28,901 swissprot, 1,073 genbank, 239,055 none
```

Protein coverage is the thing to plan around: **only 30% of rows cite a
protein at all**. 15,508 distinct accessions are cited and 14,042 resolve
(91%) — the missing 9% are entries deleted from UniProt since the 2023 build.

## Bridge to the GEMs

**EC codes.** 670 of iML1515's 917 and 850 of Human-GEM's 1,139 are present in
EnzymeMap; 247 and 289 respectively are not. So roughly three quarters of each
model's enzymatic reactions have a chemistry reference here.

**Molecules — and this is where the naive comparison misleads.**

| | exact InChIKey | skeleton block |
|---|---|---|
| iML1515 metabolites found in EnzymeMap | 202 / 969 (21%) | **676 / 969 (70%)** |
| Human-GEM metabolites found in EnzymeMap | 603 / 2,809 (21%) | **1,329 / 2,809 (47%)** |

An InChIKey's second and third blocks encode stereochemistry, isotopes and
**protonation**. A GEM stores `atp_c` as ATP(4−) at physiological pH; BRENDA
stores neutral ATP. Same molecule, different full key. Matching on the first
block — the molecular skeleton — is the standard "same compound ignoring
charge and stereo" comparison, and it more than triples the overlap.

`species.csv` carries both columns (`*_node_exact` and `*_node_skeleton`), so
you can pick per question: use the exact key when stereochemistry is part of
what you are asking, the skeleton otherwise. Do not read the exact number as
the coverage figure — it is measuring protonation conventions.

The bridge is on InChIKey rather than on SMILES strings for the same class of
reason: the same molecule is written differently by different databases, and
a string comparison silently misses most true matches. Everything is
canonicalised through rdkit first.

## Cost notes

Deduplicating SMILES **strings** before canonicalising is what makes the
molecule pass affordable: the 349,458 reactions hold 1,529,230 participants
but only 43,718 distinct strings, so rdkit parses each molecule once instead
of ~35 times. All 43,718 parse; none are dropped.

`--extras` additionally downloads `raw_reactions.csv.gz` (pre-validation, for
provenance) and `compound_to_smiles.json` (BRENDA compound name → SMILES).
Neither is needed for anything above.
