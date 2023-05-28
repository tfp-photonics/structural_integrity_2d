# vim: set fileencoding=utf-8

import h5py
import numpy as np
import pandas as pd


class HDFBrowser:
    def __init__(self, fname, mode="r"):
        self.fname = fname
        self.mode = mode

    def to_dict(self):
        f = h5py.File(self.fname, "r")
        odict = {}
        for key in f.keys():
            grp = f[key]
            grp_dict = {}
            for gkey, gval in grp.items():
                if gkey == "options":
                    grp_dict[gkey] = eval(str(gval[()]))
                elif gkey == "timestamp":
                    ts = str(gval[()]).replace("_", " ")
                    grp_dict[gkey] = pd.Timestamp(ts)
                elif gval.dtype == object:
                    grp_dict[gkey] = str(gval[()])
                else:
                    v = np.array(gval)
                    if v.size == 1:
                        grp_dict[gkey] = v.item()
                    else:
                        grp_dict[gkey] = np.array(gval)
            odict[key] = grp_dict
        f.close()
        return odict

    def to_df(self, merge=True):
        runs = []
        for run in self.to_dict().values():
            if merge:
                run.update(run["options"])
                run.pop("options", None)
            runs.append(run)
        return pd.DataFrame(runs).set_index("timestamp").sort_index()

    def __str__(self):
        f = h5py.File(self.fname, "r")
        summary = f"HDF5 file {self.fname} with {len(f.keys())} groups:\n"
        groups = "\n".join([f"  {key}" for key in f])
        f.close()
        return summary + groups
