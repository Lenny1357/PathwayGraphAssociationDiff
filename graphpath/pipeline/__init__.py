"""Experiment pipelines: the notebooks as reusable, resumable stages.

    train    graphpath.pipeline.compare.CompareExperiment     -> outputs/runs/<run>/
    score    graphpath.pipeline.score.ScoreRun                -> outputs/analysis/<run>/scores.json
    analyse  graphpath.pipeline.trajectories / analysis       -> outputs/analysis/<run>/*.json
    tables   graphpath.pipeline.tables.write_tables           -> outputs/analysis/<run>/tables/*.csv
    figures  graphpath.pipeline.figures.draw_all              -> outputs/analysis/<run>/figures/
    conditioning / signals                                    the other two experiment families

Every stage reads what the previous one wrote to disk, so each can be re-run
on its own; `scripts/` holds the command-line entry points.
"""

from graphpath.pipeline.common import ANALYSIS, DatasetBundle, RunLayout, dump_json, key, load_bundle, load_json

__all__ = ["ANALYSIS", "DatasetBundle", "RunLayout", "dump_json", "key", "load_bundle", "load_json"]
