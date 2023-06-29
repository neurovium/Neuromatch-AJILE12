# Streaming
import fsspec
from fsspec.implementations.caching import CachingFileSystem

# Numerical
import numpy as np
import natsort
import matplotlib.pyplot as plt
from matplotlib import gridspec
import h5py

# General
from tqdm import tqdm

# DANDI/NWB
from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient

def identify_elecs(group_names):
    """Determine surface v. depth ECoG electrodes"""
    is_surf = []
    for label in group_names:
        if "grid" in label.lower():
            is_surf.append(True)
        elif label.lower() in ["mhd", "latd", "lmtd", "ltpd"]:
            is_surf.append(True)  # special cases
        elif (label.lower() == "ahd") & ("PHD" not in group_names):
            is_surf.append(True)  # special case
        elif "d" in label.lower():
            is_surf.append(False)
        else:
            is_surf.append(True)
    return np.array(is_surf)

def plot_ecog_descript(
    n_elecs_tot,
    n_elecs_good,
    part_ids,
    nparts=12,
    allLH=False,
    nrows=3,
    chan_labels="all",
    width=7,
    height=3,
):
    """Plot ECoG electrode positions and identified noisy
    electrodes side by side."""
    with DandiAPIClient() as client:
        paths = []
        for file in client.get_dandiset("000055", "draft").get_assets_with_path_prefix(""):
            paths.append(file.path)
    paths = natsort.natsorted(paths)

    fig = plt.figure(figsize=(width * 3, height * 3), dpi=150)
    # First subplot: electrode locations
    ncols = nparts // nrows
    gs = gridspec.GridSpec(
        nrows=nrows,
        ncols=ncols,
        figure=fig,
        width_ratios=[width / ncols] * ncols,
        height_ratios=[height / nrows] * nrows,
        wspace=0,
        hspace=-0.5,
    )
    ax = [None] * nparts

    for part_ind in tqdm(range(nparts)):
        # Load NWB data file
        fids = [val for val in paths if "sub-" + str(part_ind + 1).zfill(2) in val]
        with DandiAPIClient() as client:
            asset = client.get_dandiset("000055", "draft").get_asset_by_path(fids[0])
            s3_path = asset.get_content_url(follow_redirects=1, strip_query=True)

        fs = CachingFileSystem(
            fs=fsspec.filesystem("http")
        )

        f = fs.open(s3_path, "rb")
        file = h5py.File(f)
        with fsspec.open(s3_path, 'rb') as s3_file:
            with NWBHDF5IO(file=file, mode='r', load_namespaces=True) as io:
                nwb = io.read()

                # Determine hemisphere to display
                if allLH:
                    sides_2_display = "l"
                else:
                    average_xpos_sign = np.nanmean(nwb.electrodes["x"][:])
                    sides_2_display = "r" if average_xpos_sign > 0 else "l"

                # Run electrode plotting function
                ax[part_ind] = fig.add_subplot(gs[part_ind // ncols, part_ind % ncols])
                plot_ecog_electrodes_mni_from_nwb_file(
                    nwb,
                    chan_labels,
                    num_grid_chans=64,
                    node_size=50,
                    colors="silver",
                    alpha=0.9,
                    sides_2_display=sides_2_display,
                    node_edge_colors="k",
                    edge_linewidths=1.5,
                    ax_in=ax[part_ind],
                    allLH=allLH,
                )

    #         ax[part_ind].text(-0.2,0.1,'P'+str(part_ind+1).zfill(2), fontsize=8)
    #     fig.text(0.1, 0.91, '(a) ECoG electrode positions', fontsize=10)

    # Second subplot: noisy electrodes per participant
    #     ax[-1] = fig.add_subplot(gs[:, -1])
    #     ax[-1].bar(part_ids,n_elecs_tot,color='lightgrey')
    #     ax[-1].bar(part_ids,n_elecs_good,color='dimgrey')
    #     ax[-1].spines['right'].set_visible(False)
    #     ax[-1].spines['top'].set_visible(False)
    #     ax[-1].set_xticklabels(part_ids, rotation=45)
    #     ax[-1].legend(['Total','Good'], frameon=False, fontsize=8)
    #     ax[-1].tick_params(labelsize=9)
    #     ax[-1].set_ylabel('Number of electrodes', fontsize=9, labelpad=0)
    #     ax[-1].set_title('(b) Total/good electrodes per participant',
    #                     fontsize=10)

    plt.show()
    return fig

def plot_ecog_electrodes_mni_from_nwb_file(
    nwb_dat,
    chan_labels="all",
    num_grid_chans=64,
    colors=None,
    node_size=50,
    figsize=(16, 6),
    sides_2_display="auto",
    node_edge_colors=None,
    alpha=0.5,
    edge_linewidths=3,
    ax_in=None,
    rem_zero_chans=False,
    allLH=False,
    zero_rem_thresh=0.99,
    elec_col_suppl=None,
):
    """
    Plots ECoG electrodes from MNI coordinate file (only for specified labels)
    NOTE: If running in Jupyter, use '%matplotlib inline' instead of '%matplotlib notebook'
    """
    # Load channel locations
    chan_info = nwb_dat.electrodes.to_dataframe()

    # Create dataframe for electrode locations
    if chan_labels == "all":
        locs = chan_info.loc[:, ["x", "y", "z"]]
    elif chan_labels == "allgood":
        locs = chan_info.loc[:, ["x", "y", "z", "good"]]
    else:
        locs = chan_info.loc[chan_labels, ["x", "y", "z"]]
    if colors is not None:
        if (locs.shape[0] > len(colors)) & isinstance(colors, list):
            locs = locs.iloc[: len(colors), :]
    #     locs.rename(columns={'X':'x','Y':'y','Z':'z'}, inplace=True)
    chan_loc_x = chan_info.loc[:, "x"].values

    # Remove NaN electrode locations (no location info)
    nan_drop_inds = np.nonzero(np.isnan(chan_loc_x))[0]
    locs.dropna(axis=0, inplace=True)  # remove NaN locations
    if (colors is not None) & isinstance(colors, list):
        colors_new, loc_inds_2_drop = [], []
        for s, val in enumerate(colors):
            if not (s in nan_drop_inds):
                colors_new.append(val)
            else:
                loc_inds_2_drop.append(s)
        colors = colors_new.copy()

        if elec_col_suppl is not None:
            loc_inds_2_drop.reverse()  # go from high to low values
            for val in loc_inds_2_drop:
                del elec_col_suppl[val]

    if chan_labels == "allgood":
        goodChanInds = chan_info.loc[:, "good", :]
        inds2drop = np.nonzero(locs["good"] == 0)[0]
        locs.drop(columns=["good"], inplace=True)
        locs.drop(locs.index[inds2drop], inplace=True)

        if colors is not None:
            colors_new, loc_inds_2_drop = [], []
            for s, val in enumerate(colors):
                if not (s in inds2drop):
                    #                     np.all(s!=inds2drop):
                    colors_new.append(val)
                else:
                    loc_inds_2_drop.append(s)
            colors = colors_new.copy()

            if elec_col_suppl is not None:
                loc_inds_2_drop.reverse()  # go from high to low values
                for val in loc_inds_2_drop:
                    del elec_col_suppl[val]

    if rem_zero_chans:
        # Remove channels with zero values (white colors)
        colors_new, loc_inds_2_drop = [], []
        for s, val in enumerate(colors):
            if np.mean(val) < zero_rem_thresh:
                colors_new.append(val)
            else:
                loc_inds_2_drop.append(s)
        colors = colors_new.copy()
        locs.drop(locs.index[loc_inds_2_drop], inplace=True)

        if elec_col_suppl is not None:
            loc_inds_2_drop.reverse()  # go from high to low values
            for val in loc_inds_2_drop:
                del elec_col_suppl[val]

    # Decide whether to plot L or R hemisphere based on x coordinates
    if len(sides_2_display) > 1:
        N, axes, sides_2_display = _setup_subplot_view(locs, sides_2_display, figsize)
    else:
        N = 1
        axes = ax_in
        if allLH:
            average_xpos_sign = np.mean(np.asarray(locs["x"]))
            if average_xpos_sign > 0:
                locs["x"] = -locs["x"]
            sides_2_display = "l"

    if colors is None:
        colors = list()

    # Label strips/depths differently for easier visualization (or use defined color list)
    if len(colors) == 0:
        for s in range(locs.shape[0]):
            if s >= num_grid_chans:
                colors.append("r")
            else:
                colors.append("b")

    if elec_col_suppl is not None:
        colors = elec_col_suppl.copy()

    # Rearrange to plot non-grid electrode first
    if num_grid_chans > 0:  # isinstance(colors, list):
        locs2 = locs.copy()
        locs2["x"] = np.concatenate(
            (locs["x"][num_grid_chans:], locs["x"][:num_grid_chans]), axis=0
        )
        locs2["y"] = np.concatenate(
            (locs["y"][num_grid_chans:], locs["y"][:num_grid_chans]), axis=0
        )
        locs2["z"] = np.concatenate(
            (locs["z"][num_grid_chans:], locs["z"][:num_grid_chans]), axis=0
        )

        if isinstance(colors, list):
            colors2 = colors.copy()
            colors2 = colors[num_grid_chans:] + colors[:num_grid_chans]
        else:
            colors2 = colors
    else:
        locs2 = locs.copy()
        if isinstance(colors, list):
            colors2 = colors.copy()
        else:
            colors2 = colors  # [colors for i in range(locs2.shape[0])]

    # Plot the result
    _plot_electrodes(
        locs2,
        node_size,
        colors2,
        axes,
        sides_2_display,
        N,
        node_edge_colors,
        alpha,
        edge_linewidths,
    )
    
def _plot_electrodes(
    locs,
    node_size,
    colors,
    axes,
    sides_2_display,
    N,
    node_edge_colors,
    alpha,
    edge_linewidths,
    marker="o",
):
    """
    Handles plotting of electrodes.
    """
    if N == 1:
        ni_plt.plot_connectome(
            np.eye(locs.shape[0]),
            locs,
            output_file=None,
            node_kwargs={
                "alpha": alpha,
                "edgecolors": node_edge_colors,
                "linewidths": edge_linewidths,
                "marker": marker,
            },
            node_size=node_size,
            node_color=colors,
            axes=axes,
            display_mode=sides_2_display,
        )
    elif sides_2_display == "yrz" or sides_2_display == "ylz":
        colspans = [
            5,
            6,
            5,
        ]  # different sized subplot to make saggital view similar size to other two slices
        current_col = 0
        total_colspans = int(np.sum(np.asarray(colspans)))
        for ind, colspan in enumerate(colspans):
            axes[ind] = plt.subplot2grid(
                (1, total_colspans), (0, current_col), colspan=colspan, rowspan=1
            )
            ni_plt.plot_connectome(
                np.eye(locs.shape[0]),
                locs,
                output_file=None,
                node_kwargs={
                    "alpha": alpha,
                    "edgecolors": node_edge_colors,
                    "linewidths": edge_linewidths,
                    "marker": marker,
                },
                node_size=node_size,
                node_color=colors,
                axes=axes[ind],
                display_mode=sides_2_display[ind],
            )
            current_col += colspan
    else:
        for i in range(N):
            ni_plt.plot_connectome(
                np.eye(locs.shape[0]),
                locs,
                output_file=None,
                node_kwargs={
                    "alpha": alpha,
                    "edgecolors": node_edge_colors,
                    "linewidths": edge_linewidths,
                    "marker": marker,
                },
                node_size=node_size,
                node_color=colors,
                axes=axes[i],
                display_mode=sides_2_display[i],
            )
                