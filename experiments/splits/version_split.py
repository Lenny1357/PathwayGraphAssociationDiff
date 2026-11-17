"""Train on an older Human-GEM, test what the model puts back.

The idea is the retrospective version of a held-out set: instead of hiding
edges at random, hide the ones the curators had not added yet. Anything the
model proposes that later turned out to be real is evidence it generalises to
biology rather than to this release's quirks.

WHAT IS AND IS NOT PREDICTABLE
------------------------------
A new release adds new NODES and new STRUCTURE, and only the second is
predictable by a model trained on the old one:

    v1.13.0 -> current      178 new reactions, 177 new metabolites, 46 new genes

A new metabolite has an id the old vocabulary does not contain, so the model
cannot emit it — not because it failed, but because the token does not exist.
The same is true of new reactions: `MAR20168` is itself a new id. So "how many
newly added nodes can we predict back" has no well-posed answer with a fixed
vocabulary.

What a new reaction DOES create, that an old-vocabulary model can express, is
new CO-MEMBERSHIP among entities that already existed: metabolites that now
share a reaction and did not before, genes that now share one. That is a real,
well-powered target:

    gained associations   3,066   (gene-gene 1,614, gene-met 962, met-met 490)
    lost associations     9,824   <- curators REMOVED these; a good model
                                     should not propose them, so they are a
                                     free negative control

Because the target is a set of entity pairs, the evaluation is exactly the
link-prediction harness in edge_split.py — `to_edge_split()` hands it over.
Nothing is duplicated.

If you want new nodes to be predictable at all, the vocabulary has to carry
information the model can generalise over — sequence embeddings for genes,
structure for metabolites — so that a never-trained token still has a
meaningful embedding. Without that, an unseen token's logit is its random
initialisation and the question measures nothing.
"""

import io
import itertools
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

RAW = "https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/{tag}/model/Human-GEM.mat"


class Release:
    """One Human-GEM release, reduced to what a split needs."""

    def __init__(self, tag, rxns, mets, genes, S, RG, eccodes=None):
        self.tag, self.rxns, self.mets, self.genes = tag, rxns, mets, genes
        self.S, self.RG, self.eccodes = S, RG, eccodes

    def __repr__(self):
        return (f"Release({self.tag}: {len(self.rxns):,} rxns, "
                f"{len(self.mets):,} mets, {len(self.genes):,} genes)")

    @staticmethod
    def _strs(v):
        out = []
        for x in np.asarray(v, dtype=object).ravel():
            out.append(x if isinstance(x, str)
                       else ("" if getattr(x, "size", 1) == 0 else str(np.ravel(x)[0])))
        return np.array(out, dtype=object)

    @classmethod
    def from_mat(cls, path_or_bytes, tag="?"):
        src = (io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes)
               else str(path_or_bytes))
        m = sio.loadmat(src, struct_as_record=False, squeeze_me=True)
        g = m[[k for k in m if not k.startswith("__")][0]]
        ec = cls._strs(g.eccodes) if "eccodes" in g._fieldnames else None
        return cls(tag, cls._strs(g.rxns), cls._strs(g.mets), cls._strs(g.genes),
                   sp.csc_matrix(g.S), sp.csr_matrix(g.rxnGeneMat), ec)

    @classmethod
    def fetch(cls, tag, cache_dir="gem_cache", refresh=False):
        d = Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"Human-GEM-{tag}.mat"
        if not f.exists() or refresh:
            print(f"  downloading Human-GEM {tag} ...", flush=True)
            f.write_bytes(urllib.request.urlopen(RAW.format(tag=tag), timeout=180).read())
        return cls.from_mat(f, tag)

    def co_membership(self, only: Optional[set] = None, cap: int = 40) -> set:
        """
        Unordered pairs of entities that share a reaction.

        `cap` drops pseudo-reactions with dozens of participants (biomass and
        the like). They would contribute a dense clique that swamps every real
        association and is not a biological claim about any pair.
        """
        out = set()
        for j in range(self.S.shape[1]):
            ms = [self.mets[i] for i in self.S.indices[self.S.indptr[j]:self.S.indptr[j+1]]]
            gs = [self.genes[i] for i in self.RG.indices[self.RG.indptr[j]:self.RG.indptr[j+1]]]
            ent = sorted(set(ms) | set(gs))
            if only is not None and not all(e in only for e in ent):
                continue
            if len(ent) > cap:
                continue
            out.update(itertools.combinations(ent, 2))
        return out


class VersionSplit:
    """
    What changed between two releases, in the form the model can be asked about.

    gained : associations present in `new` and absent from `old`, over entities
             BOTH releases contain. The prediction target.
    lost   : the reverse. Curators removed these, so proposing them is a
             (mild) negative — mild because a removal can be a re-annotation
             rather than a refutation.
    """

    def __init__(self, old: Release, new: Release, cap: int = 40):
        self.old, self.new = old, new
        shared = (set(old.mets) & set(new.mets)) | (set(old.genes) & set(new.genes))
        o = old.co_membership(only=shared, cap=cap)
        n = new.co_membership(only=shared, cap=cap)
        self.gained, self.lost = n - o, o - n
        self.shared_entities = shared

        # Where each gained association comes from. This split matters more
        # than it looks: 99% of them come from curators EDITING the gene rule
        # of a reaction that already existed, not from new chemistry. Those
        # are different claims — "can it predict a re-annotation" versus "can
        # it predict a reaction nobody had written down" — and mixing them
        # lets the easy one hide the hard one.
        new_rxn_ids = set(new.rxns) - set(old.rxns)
        self.gained_from_new_reactions, self.gained_from_edits = set(), set()
        for j in range(new.S.shape[1]):
            ms = [new.mets[i] for i in new.S.indices[new.S.indptr[j]:new.S.indptr[j+1]]]
            gs = [new.genes[i] for i in new.RG.indices[new.RG.indptr[j]:new.RG.indptr[j+1]]]
            ent = sorted(set(ms) | set(gs))
            if len(ent) > cap or not all(e in shared for e in ent):
                continue
            bucket = (self.gained_from_new_reactions if new.rxns[j] in new_rxn_ids
                      else self.gained_from_edits)
            for pr in itertools.combinations(ent, 2):
                if pr in self.gained:
                    bucket.add(pr)
        self.gained_from_edits -= self.gained_from_new_reactions
        self.added_nodes = {
            "reactions": sorted(set(new.rxns) - set(old.rxns)),
            "metabolites": sorted(set(new.mets) - set(old.mets)),
            "genes": sorted(set(new.genes) - set(old.genes))}
        self.dropped_nodes = {
            "reactions": sorted(set(old.rxns) - set(new.rxns)),
            "metabolites": sorted(set(old.mets) - set(new.mets)),
            "genes": sorted(set(old.genes) - set(new.genes))}

    @staticmethod
    def _t(e):
        return "gene" if str(e).startswith("ENSG") else "metabolite"

    def summary(self) -> str:
        L = [f"VersionSplit {self.old.tag} -> {self.new.tag}",
             f"  entities in both releases      {len(self.shared_entities):,}"]
        for k in ("reactions", "metabolites", "genes"):
            L.append(f"  new {k:<26}{len(self.added_nodes[k]):>7,}   "
                     f"dropped {len(self.dropped_nodes[k]):,}")
        L.append(f"  GAINED associations            {len(self.gained):>7,}   "
                 f"<- the prediction target")
        L.append(f"    of which, new chemistry      {len(self.gained_from_new_reactions):>7,}   "
                 f"<- from reactions that did not exist")
        L.append(f"    of which, re-annotation      {len(self.gained_from_edits):>7,}   "
                 f"<- gene rules edited on existing reactions")
        L.append(f"  LOST associations              {len(self.lost):>7,}   "
                 f"<- negative control")
        for lbl, S_ in (("gained", self.gained), ("lost", self.lost)):
            c = Counter(tuple(sorted((self._t(a), self._t(b)))) for a, b in S_)
            L.append(f"    {lbl:<8}" + "  ".join(f"{k[0]}-{k[1]} {v:,}"
                                                 for k, v in c.most_common()))
        L.append("  new NODES are reported, not predicted: a token absent from the")
        L.append("  training vocabulary has no probability to measure.")
        return "\n".join(L)

    # ── hand over to the edge-split evaluation ─────────────────────────────

    def to_edge_split(self, dataset, node_type_fn, which="gained", pad_id=0):
        """which: "gained" | "new_chemistry" | "re_annotation" | "lost"."""
        """
        Re-express the changed associations as an `EdgeSplit`-compatible
        object in the DATASET's token space, so `link_prediction` and
        `MaskedEdgeDataset` work on it unchanged.

        Only pairs whose entities both appear in the dataset vocabulary
        survive; the count is reported so a poor overlap is visible rather
        than silently shrinking the test.
        """
        from edge_split import EdgeSplit, canon

        tok_of = {}
        for tok, name in dataset.id_to_node.items():
            tok_of.setdefault(str(name), tok)
            if str(name).startswith("rxn_"):
                tok_of.setdefault(str(name)[4:], tok)

        target = {"gained": self.gained,
                  "new_chemistry": self.gained_from_new_reactions,
                  "re_annotation": self.gained_from_edits,
                  "lost": self.lost}[which]
        held, missing = set(), 0
        for a, b in target:
            ta, tb = tok_of.get(a), tok_of.get(b)
            if ta is None or tb is None or ta == tb:
                missing += 1
                continue
            held.add(canon(ta, tb))

        sp_ = EdgeSplit.__new__(EdgeSplit)
        sp_.dataset, sp_.pad_id = dataset, pad_id
        sp_.type_of_token = {t: node_type_fn(n) for t, n in dataset.id_to_node.items()}
        sp_.held_out = held
        sp_.n_eligible = len(target)
        sp_.n_edges_total = len(target)
        sp_.n_skipped_min_degree = 0
        sp_.held_by_type = Counter(
            tuple(sorted((sp_.type_of_token.get(a, "?"), sp_.type_of_token.get(b, "?"))))
            for a, b in held)
        sp_.n_not_in_vocab = missing
        return sp_
