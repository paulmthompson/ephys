"""Ad-hoc preview: scatter marginals with data-coordinate diff inset."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ephys.plotting.scatter_marginals import (
    ScatterMarginalsData,
    ScatterMarginalsOptions,
    draw_scatter_marginals_into,
)


def main() -> None:
    rng = np.random.default_rng(42)
    n = 80
    x = rng.uniform(2.0, 18.0, size=n)
    y = x + rng.normal(0.0, 2.5, size=n)

    data = ScatterMarginalsData(x=x, y=y)
    options = ScatterMarginalsOptions(
        xlabel="Mean rate A (Hz)",
        ylabel="Mean rate B (Hz)",
        marginal_hist_bins=15,
        diff_hist_bins=8,
        diff_inset_center=20.0,
        diff_inset_d_half_extent=5.0,
        diff_inset_xtick_labels=False,
        diff_inset_count_extent=None,
        point_size=28.0,
        point_edgecolor="0.25",
        scatter_lim=(0.0, 25.0),
    )

    fig = plt.figure(figsize=(5.5, 5.5))
    cell = fig.add_gridspec(1, 1)[0, 0]
    draw_scatter_marginals_into(fig, cell, data, options)
    fig.suptitle("Scatter marginals preview (random data)", fontsize=11, y=0.98)

    out = "scatter_marginals_preview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
