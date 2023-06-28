# Numerical
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_ecog_pow(
    lp,
    rois_plt,
    freq_range,
    sbplt_titles,
    part_id="P01",
    n_parts=12,
    nrows=2,
    ncols=4,
    figsize=(7, 4),
):
    """Plot ECoG projected spectral power."""
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, dpi=150)

    # Plot projected power for all participants
    fig, ax = _ecog_pow_group(
        fig,
        ax,
        lp,
        rois_plt,
        freq_range,
        sbplt_titles,
        n_parts,
        nrows,
        ncols,
        row_ind=0,
    )

    # Plot projected power for 1 participant
    fig, ax = _ecog_pow_single(
        fig,
        ax,
        lp,
        rois_plt,
        freq_range,
        sbplt_titles,
        n_parts,
        nrows,
        ncols,
        row_ind=1,
        part_id=part_id,
    )

    fig.tight_layout()
    plt.show()


def _ecog_pow_group(
    fig,
    ax,
    lp,
    rois_plt,
    freq_range,
    sbplt_titles,
    n_parts=12,
    nrows=2,
    ncols=4,
    row_ind=0,
):
    """Plot projected power for all participants."""
    freqs_vals = np.arange(freq_range[0], freq_range[1] + 1).tolist()
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.1)
    power, freqs, parts = [], [], []
    n_wins_sbj = []
    for k, roi in enumerate(rois_plt):
        power_roi, freqs_roi, parts_roi = [], [], []
        for j in range(n_parts):
            dat = np.load(lp + "P" + str(j + 1).zfill(2) + "_" + roi + ".npy")
            dat = 10 * np.log10(dat)
            for i in range(dat.shape[0]):
                power_roi.extend(dat[i, :].tolist())
                freqs_roi.extend(freqs_vals)
                parts_roi.extend(["P" + str(j + 1).zfill(2)] * len(freqs_vals))
            if k == 0:
                n_wins_sbj.append(dat.shape[0])
        power.extend(power_roi)
        freqs.extend(freqs_roi)
        parts.extend(parts_roi)

        parts_uni = np.unique(np.asarray(parts_roi))[::-1].tolist()
        df_roi = pd.DataFrame(
            {"Power": power_roi, "Freqs": freqs_roi, "Parts": parts_roi}
        )
        col = k % ncols
        ax_curr = ax[row_ind, col] if nrows > 1 else ax[col]
        leg = False  # 'brief' if k==3 else False
        sns.lineplot(
            data=df_roi,
            x="Freqs",
            y="Power",
            hue="Parts",
            ax=ax_curr,
            ci="sd",
            legend=leg,
            palette=["darkgray"] * len(parts_uni),
            hue_order=parts_uni,
        )  # palette='Blues'
        #     ax_curr.set_xscale('log')
        ax_curr.set_xlim(freq_range)
        ax_curr.set_ylim([-20, 30])
        ax_curr.spines["right"].set_visible(False)
        ax_curr.spines["top"].set_visible(False)
        ax_curr.set_xlim(freq_range)
        ax_curr.set_xticks(
            [freq_range[0]] + np.arange(20, 101, 20).tolist() + [freq_range[1]]
        )
        ylab = ""  # '' if k%ncols > 0 else 'Power\n(dB)'  # 10log(uV^2)
        xlab = ""  # 'Frequency (Hz)' if k//ncols==(nrows-1) else ''
        ax_curr.set_ylabel(ylab, rotation=0, labelpad=15, fontsize=9)
        ax_curr.set_xlabel(xlab, fontsize=9)
        if k % ncols > 0:
            l_yticks = len(ax_curr.get_yticklabels())
            ax_curr.set_yticks(ax_curr.get_yticks().tolist())
            ax_curr.set_yticklabels([""] * l_yticks)
        ax_curr.tick_params(axis="both", which="major", labelsize=8)
        ax_curr.set_title(sbplt_titles[k], fontsize=9)
    return fig, ax


def _ecog_pow_single(
    fig,
    ax,
    lp,
    rois_plt,
    freq_range,
    sbplt_titles,
    n_parts=12,
    nrows=2,
    ncols=4,
    row_ind=1,
    part_id="P01",
):
    """Plot projected power for a single participant."""
    part_id = "P01"
    freqs_vals = np.arange(freq_range[0], freq_range[1] + 1).tolist()
    power, freqs, parts = [], [], []
    n_wins_sbj = []
    for k, roi in enumerate(rois_plt):
        power_roi, freqs_roi, parts_roi = [], [], []

        dat = np.load(lp + part_id + "_" + roi + ".npy")
        dat = 10 * np.log10(dat)
        for i in range(dat.shape[0]):
            power_roi.extend(dat[i, :].tolist())
            freqs_roi.extend(freqs_vals)
            parts_roi.extend([i] * len(freqs_vals))
        if k == 0:
            n_wins_sbj.append(dat.shape[0])
        power.extend(power_roi)
        freqs.extend(freqs_roi)
        parts.extend(parts_roi)

        parts_uni = np.unique(np.asarray(parts_roi))[::-1].tolist()
        df_roi = pd.DataFrame(
            {"Power": power_roi, "Freqs": freqs_roi, "Parts": parts_roi}
        )
        col = k % ncols
        ax_curr = ax[row_ind, col] if nrows > 1 else ax[col]
        leg = False  # 'brief' if k==3 else False
        sns.lineplot(
            data=df_roi,
            x="Freqs",
            y="Power",
            hue="Parts",
            ax=ax_curr,
            ci=None,
            legend=leg,
            palette=["darkgray"] * len(parts_uni),
            hue_order=parts_uni,
            linewidth=0.2,
        )  # palette='Blues'
        ax_curr.set_xlim(freq_range)
        ax_curr.set_ylim([-20, 30])
        ax_curr.spines["right"].set_visible(False)
        ax_curr.spines["top"].set_visible(False)
        ax_curr.set_xlim(freq_range)
        ax_curr.set_xticks(
            [freq_range[0]] + np.arange(20, 101, 20).tolist() + [freq_range[1]]
        )
        ylab = ""  # '' if k%ncols > 0 else 'Power\n(dB)'  # 10log(uV^2)
        xlab = ""  # 'Frequency (Hz)' if k//ncols==(nrows-1) else ''
        ax_curr.set_ylabel(ylab, rotation=0, labelpad=15, fontsize=9)
        ax_curr.set_xlabel(xlab, fontsize=9)
        if k % ncols > 0:
            l_yticks = len(ax_curr.get_yticklabels())
            ax_curr.set_yticks(ax_curr.get_yticks().tolist())
            ax_curr.set_yticklabels([""] * l_yticks)
        ax_curr.tick_params(axis="both", which="major", labelsize=8)
        ax_curr.set_title(sbplt_titles[k], fontsize=9)
    return fig, ax