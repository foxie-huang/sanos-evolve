"""Path roots for the public repo.

`DATA` is the artifacts directory shipped here -- fit records and result artifacts in one place,
because every consumer resolves inputs as os.path.join(_paths.DATA, ...). `FIGS` is where generated
tables and figures land.
"""
import os
import sys

V3 = os.path.dirname(os.path.abspath(__file__))          # <repo>/v3
ROOT = os.path.dirname(V3)                               # <repo>
DATA = os.path.join(V3, "artifacts")                     # fit records + result artifacts
FIGS = os.path.join(ROOT, "figs_v3")                     # generated tables/figures

for _p in (os.path.join(V3, "kernel_fast"), os.path.join(V3, "sanos"),
           os.path.join(V3, "diagnostics"), os.path.join(V3, "figures"),
           os.path.join(ROOT, "data"), os.path.join(ROOT, "poc"),
           os.path.join(ROOT, "scripts"), DATA):
    if _p not in sys.path:
        sys.path.append(_p)
for _p in (os.path.join(V3, "kernel_fast"), os.path.join(V3, "sanos")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
