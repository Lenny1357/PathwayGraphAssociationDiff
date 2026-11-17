"""Currency metabolites and common cofactors, excluded from co-expression evidence routes.

Hub metabolites such as water, protons and ATP take part in a large share of all
reactions, so in the metabolic network they put almost any two entities within two
hops of each other. Evidence propagated through them (an entity reaches a gene via
H2O) says nothing about a specific biochemical relationship. The co-expression
analyses (`coexpr_support`, `discovery`) therefore drop these entities: as endpoints
of a scored link, as intermediate steps of a support route, and from the random
control, so generated links and the control are scored on equal terms.

The list is explicit and short on purpose: classic currency metabolites (water,
protons, gases, phosphate), nucleotide energy carriers, redox cofactors, coenzyme A,
quinones, methyl-group carriers and common inorganic ions. Substrates that also act
as carriers in some reactions (acetyl-CoA, glutamate, 2-oxoglutarate, glutathione,
amino acids) are kept, because links involving them can be pathway-specific.

Matching is by compartment-free identifier root (BiGG ids for iML1515, e.g. `atp_c`
-> `atp`) or by common name (Human-GEM, e.g. `MAM01371c` -> "ATP"), case-insensitive.
"""

import re

# BiGG identifier roots (iML1515 and other BiGG models)
BIGG_ROOTS = frozenset({
    "h", "h2o", "o2", "co2", "pi", "ppi", "nh4", "hco3", "h2o2",
    "atp", "adp", "amp", "gtp", "gdp", "gmp", "utp", "udp", "ump", "ctp", "cdp", "cmp", "itp", "idp", "imp",
    "datp", "dadp", "damp", "dgtp", "dgdp", "dctp", "dcdp", "dttp", "dtdp", "dutp",
    "nad", "nadh", "nadp", "nadph", "fad", "fadh2", "fmn", "fmnh2", "coa",
    "q8", "q8h2", "mqn8", "mql8", "2dmmq8", "2dmmql8",
    "amet", "ahcys",
    "na1", "k", "cl", "fe2", "fe3", "ca2", "mg2", "zn2", "mn2", "cu2", "cobalt2",
})

# common names (Human-GEM and other models), compared lower-case without spaces
NAMES = frozenset(n.lower().replace(" ", "") for n in (
    "H+", "H2O", "O2", "CO2", "Pi", "PPi", "phosphate", "diphosphate", "NH3", "NH4+", "ammonia", "ammonium",
    "HCO3-", "bicarbonate", "H2O2", "hydrogen peroxide",
    "ATP", "ADP", "AMP", "GTP", "GDP", "GMP", "UTP", "UDP", "UMP", "CTP", "CDP", "CMP", "ITP", "IDP", "IMP",
    "dATP", "dADP", "dAMP", "dGTP", "dGDP", "dCTP", "dCDP", "dTTP", "dTDP", "dUTP",
    "NAD+", "NADH", "NADP+", "NADPH", "FAD", "FADH2", "FMN", "FMNH2", "CoA", "coenzyme A",
    "ubiquinone", "ubiquinol", "menaquinone", "menaquinol",
    "SAM", "SAH", "S-adenosyl-L-methionine", "S-adenosyl-L-homocysteine",
    "Na+", "K+", "Cl-", "Fe2+", "Fe3+", "Ca2+", "Mg2+", "Zn2+", "Mn2+", "Cu2+",
))

_BIGG_COMPARTMENT = re.compile(r"_[a-z]{1,2}$")


def is_cofactor(name: str, common: str = "") -> bool:
    """True for a currency metabolite or common cofactor, by BiGG root or common name."""
    root = _BIGG_COMPARTMENT.sub("", str(name)).lower()
    if root in BIGG_ROOTS:
        return True
    return str(common).lower().replace(" ", "") in NAMES


def cofactor_ids(bundle) -> set:
    """Token ids of every cofactor in a dataset bundle's vocabulary (metabolites only)."""
    out = set()
    for i, n in bundle.id_to_node.items():
        if bundle.ntype(n) == "metabolite" and is_cofactor(n, bundle.common(n)):
            out.add(int(i))
    return out
