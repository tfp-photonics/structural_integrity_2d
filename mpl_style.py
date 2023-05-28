import matplotlib

matplotlib.use("pgf")

import matplotlib.pyplot as plt

pgf_config = {
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "pgf.rcfonts": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica Neue",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.titlepad": 0,
    "axes.labelsize": 14,
    "scatter.marker": "x",
    "lines.markersize": 4,
    "lines.linewidth": 1,
    "pgf.preamble": [
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{graphicx}",
        r"\usepackage{amsmath}",
        r"\usepackage{sansmathfonts}",
        r"\usepackage[scaled=0.85]{helvet}",
        r"\renewcommand{\rmdefault}{\sfdefault}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[math]{blindtext}",
        r"\usepackage{physics}",
        r"\usepackage{siunitx}",
        r"\sisetup{output-exponent-marker=\text{e},exponent-product={},retain-explicit-plus}",
    ],
    "savefig.pad_inches": 0,
    "savefig.transparent": True,
}
plt.rcParams.update(pgf_config)
