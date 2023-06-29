# # Streaming
# import fsspec
# from fsspec.implementations.caching import CachingFileSystem

# fs = CachingFileSystem(
#     fs=fsspec.filesystem("http")
# )

# Numerical 
import natsort
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import h5py
from scipy.signal import medfilt
from nilearn import plotting as ni_plt

# General
from tqdm import tqdm

# DANDI/NWB
from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient
from nwbwidgets.utils.timeseries import align_by_times, timeseries_time_to_ind
import ndx_events

def clabel_table_create(
    common_acts, n_parts=12, data_lp="/data2/users/stepeter/files_nwb/downloads/000055/", fs=None
):
    """Create table of coarse label durations across participants.
    Labels to include in the table are specified by common_acts."""
    with DandiAPIClient() as client:
        paths = []
        for file in client.get_dandiset("000055", "draft").get_assets_with_path_prefix(""):
            paths.append(file.path)
    paths = natsort.natsorted(paths)

    vals_all = np.zeros([n_parts, len(common_acts) + 1])
    for part_ind in tqdm(range(n_parts)):
        fids = [val for val in paths if "sub-" + str(part_ind + 1).zfill(2) in val]
        for fid in fids:
            with DandiAPIClient() as client:
                asset = client.get_dandiset("000055", "draft").get_asset_by_path(fid)
                s3_path = asset.get_content_url(follow_redirects=1, strip_query=True)
            f = fs.open(s3_path, "rb")
            file = h5py.File(f)
            with NWBHDF5IO(file=file, mode='r', load_namespaces=True) as io:
                nwb = io.read()

                curr_labels = nwb.intervals["epochs"].to_dataframe()
                durations = (
                    curr_labels.loc[:, "stop_time"].values
                    - curr_labels.loc[:, "start_time"].values
                )

                # Add up durations of each label
                for s, curr_act in enumerate(common_acts):
                    for i, curr_label in enumerate(curr_labels["labels"].tolist()):
                        if curr_act in curr_label.split(", "):
                            vals_all[part_ind, s] += durations[i] / 3600

                # Add up total durations of selected labels (avoid double counting)
                for i, curr_label in enumerate(curr_labels["labels"].tolist()):
                    in_lab_grp = False
                    for sub_lab in curr_label.split(", "):
                        if sub_lab in common_acts:
                            in_lab_grp = True
                    vals_all[part_ind, -1] += durations[i] / 3600 if in_lab_grp else 0
            del nwb, io

    # Make final table/dataframe
    common_acts_col = [val.lstrip("Blocklist (").rstrip(")") for val in common_acts]
    df_all = pd.DataFrame(
        vals_all.round(1),
        index=["P" + str(val + 1).zfill(2) for val in range(n_parts)],
        columns=common_acts_col + ["Total"],
    )
    return df_all

def prune_clabels(
    clabels_orig, targeted=False, targ_tlims=[13, 17], first_val=True, targ_label="Eat"
):
    """Modify coarse behavior labels based on whether
    looking at whole day (targeted=False) or specific
    hours (targeted=True). When selecting specific
    hours, can look at either the first (first_val=True)
    or last (first_val=False) label if there are multiple
    overlapping activity labels."""
    clabels = clabels_orig.copy()
    if not targeted:
        for i in range(len(clabels_orig)):
            lab = clabels_orig.loc[i, "labels"]
            if lab[:5] == "Block":
                clabels.loc[i, "labels"] = "Blocklist"
            elif lab == "":
                clabels.loc[i, "labels"] = "Blocklist"
            elif lab not in ["Sleep/rest", "Inactive"]:
                clabels.loc[i, "labels"] = "Active"
    else:
        for i in range(len(clabels_orig)):
            lab = clabels_orig.loc[i, "labels"]
            if targ_label in lab.split(", "):
                clabels.loc[i, "labels"] = targ_label
            else:
                clabels.loc[i, "labels"] = "Blocklist"
    #             if lab[:5] == 'Block':
    #                 clabels.loc[i, 'labels'] = 'Blocklist'
    #             elif lab == '':
    #                 clabels.loc[i, 'labels'] = 'Blocklist'
    #             elif first_val:
    #                 clabels.loc[i, 'labels'] = lab.split(', ')[0]
    #             else:
    #                 clabels.loc[i, 'labels'] = lab.split(', ')[-1]

    if targeted:
        start_val, end_val = targ_tlims[0] * 3600, targ_tlims[1] * 3600
        clabels = clabels[
            (clabels["start_time"] >= start_val) & (clabels["stop_time"] <= end_val)
        ]
        clabels.reset_index(inplace=True)
    uni_labs = np.unique(clabels["labels"].values)
    return clabels, uni_labs


def plot_clabels(
    clabels,
    uni_labs,
    targeted=False,
    first_val=True,
    targ_tlims=[13, 17],
    scale_fact=1 / 3600,
    bwidth=0.5,
    targlab_colind=0,
):
    """Plot coarse labels for one recording day.
    Note that the colors for the plots are currently
    pre-defined to work for sub-01 day 4."""
    # Define colors for each label
    act_cols = plt.get_cmap("Reds")(np.linspace(0.15, 0.85, 5))
    if targeted:
        category_colors = np.array(["w", act_cols[targlab_colind]], dtype=object)
    #         if first_val:
    #             category_colors = np.array(['dimgray', act_cols[1], act_cols[2],
    #                                         act_cols[0], act_cols[3], act_cols[4]],
    #                                        dtype=object)
    #         else:
    #             category_colors = np.array(['dimgray', act_cols[1], act_cols[0],
    #                                         act_cols[3], act_cols[4]],
    #                                        dtype=object)
    else:
        category_colors = np.array(
            [[1, 128 / 255, 178 / 255], "dimgray", "lightgreen", "lightskyblue"],
            dtype=object,
        )

    # Plot each label as a horizontal bar
    fig, ax = plt.subplots(figsize=(20, 2), dpi=150)
    for i in range(len(uni_labs)):
        lab_inds = np.nonzero(uni_labs[i] == clabels["labels"].values)[0]
        lab_starts = clabels.loc[lab_inds, "start_time"].values
        lab_stops = clabels.loc[lab_inds, "stop_time"].values
        lab_widths = lab_stops - lab_starts
        rects = ax.barh(
            np.ones_like(lab_widths),
            lab_widths * scale_fact,
            left=lab_starts * scale_fact,
            height=bwidth,
            label=uni_labs[i],
            color=category_colors[i],
        )
    ax.legend(
        ncol=len(uni_labs), bbox_to_anchor=(0, 1), loc="lower left", fontsize="small"
    )

    # Define x-axis based on if targeted window or not
    if targeted:
        plt.xlim(targ_tlims)
        targ_tlims_int = [int(val) for val in targ_tlims]
        plt.xticks(targ_tlims_int)
        ax.set_xticklabels(
            ["{}:00".format(targ_tlims_int[0]), "{}:00".format(targ_tlims_int[-1])]
        )
    else:
        plt.xlim([0, 24])
        plt.xticks([0, 12, 24])
        ax.set_xticklabels(["0:00", "12:00", "0:00"])

    # Remove border lines and show plot
    ax.yaxis.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()
    return fig