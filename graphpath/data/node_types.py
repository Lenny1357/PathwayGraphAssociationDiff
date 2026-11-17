"""Node id -> entity type / common name, for every dataset in the project.

Each dataset prefixes its node ids differently, so the type rule is
per-dataset; pick the one matching the graphs you loaded.

    human (Human-GEM)   rxn_MAR03905 / MAM01234c / ENSG00000106009 / else enzyme
    ecoli (iML1515)     rxn_PGI / enz_b0002+b0003 / b4025 / else metabolite
    kegg                rxn_R01786 / ec_2.7.1.1 / ko_K00844 / cpd_C00267

`default_node_type` is the human rule, which the benchmarks fall back to.
`TYPE_MAP` is the integer order every trained model used for
`set_node_types()`; keep it, or the type-pair bias means something else.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

import pandas as pd

TYPE_MAP = {"gene": 0, "metabolite": 1, "enzyme": 2, "reaction": 3}
TYPE_NAMES = ["gene", "metabolite", "enzyme", "reaction"]

_BNUM = re.compile(r"^b\d{4}$")


def human_node_type(name: str) -> str:
    name = str(name)
    if name.startswith("rxn_"):
        return "reaction"
    if name.startswith("MAM"):
        return "metabolite"
    if name.startswith("ENS"):
        return "gene"
    return "enzyme"


def ecoli_node_type(name: str) -> str:
    name = str(name)
    if name.startswith("rxn_"):
        return "reaction"
    if name.startswith("enz_"):
        return "enzyme"
    if _BNUM.match(name):
        return "gene"
    return "metabolite"


def kegg_node_type(name: str) -> str:
    name = str(name)
    if name.startswith("rxn_"):
        return "reaction"
    if name.startswith(("ec_", "enz_")):
        return "enzyme"
    if name.startswith("ko_"):
        return "gene"
    return "metabolite"


NODE_TYPE_FN: dict[str, Callable[[str], str]] = {
    "human": human_node_type, "ecoli": ecoli_node_type, "kegg": kegg_node_type,
}
default_node_type = human_node_type


def node_type_fn(dataset: str) -> Callable[[str], str]:
    """The type rule for a dataset name ("human", "ecoli", "kegg")."""
    return NODE_TYPE_FN[dataset]


def load_name_map(path: str) -> dict:
    """node_name_map.csv -> {id: common_name}. Accepts an `id` or `idx` key column."""
    df = pd.read_csv(path)
    key = "id" if "id" in df.columns else "idx"
    if key not in df.columns or "common_name" not in df.columns:
        raise ValueError(f"{path} must have an 'id' (or 'idx') and a 'common_name' "
                         f"column; found {list(df.columns)}")
    return dict(zip(df[key], df["common_name"]))


def make_common_name(name_map: Optional[dict]):
    """A lookup that falls back to the raw id when a name is missing."""
    def get_common_name(node_id):
        if not name_map:
            return node_id
        val = name_map.get(node_id)
        if val is not None and pd.notna(val) and str(val):
            return val
        return node_id
    return get_common_name


make_get_common_name = make_common_name   # the name the E. coli / KEGG notebooks use
