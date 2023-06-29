# # Streaming
# import fsspec
# from fsspec.implementations.caching import CachingFileSystem

# fs = CachingFileSystem(
#     fs=fsspec.filesystem("http")
# )

# Numerical
import numpy as np
import seaborn as sns
import pandas as pd
import natsort
import h5py

# DANDI/NWB
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO


def plot_wrist_trajs(
    fig,
    ax,
    lp=None,
    base_start=-1.5,
    base_end=-1,
    before=3,
    after=3,
    fs_video=30,
    n_parts=12,
):
    """Plot contralateral wrist trajectories during move onset events."""
    df_pose, part_lst = _get_wrist_trajs(
        base_start, base_end, before, after, fs_video, n_parts
    )

    df_pose_orig = df_pose.copy()
    df_pose = df_pose_orig.loc[df_pose["Contra"] == "contra", :]

    # Set custom color palette
    sns.set_palette(sns.color_palette(["gray"]))

    uni_sbj = np.unique(np.asarray(part_lst))

    for j in range(n_parts):
        sns.lineplot(
            x="Time",
            y="Displ",
            data=df_pose[df_pose["Sbj"] == uni_sbj[j]],
            ax=ax,
            linewidth=1.5,
            hue="Contra",
            legend=False,
            estimator=np.median,
            ci=95,
        )

    ax.set_ylim([0, 60])
    ax.set_xlim([-0.5, 1.5])
    ax.set_xticks([-0.5, 0, 0.5, 1, 1.5])
    ax.set_ylabel("Displacement (px)", fontsize=9)
    ax.set_xlabel("Time (sec)", fontsize=9)
    sns.set_style("ticks")
    sns.despine()
    ax.axvline(0, linewidth=1.5, color="black", linestyle="--")
    ax.set_title("(b) Contralateral wrist trajectories during move events", fontsize=10)


def _get_wrist_trajs(
    base_start=-1.5, base_end=-1, before=3, after=3, fs_video=30, n_parts=12, fs=None
):
    """Load in wrist trajectories around move onset events."""
    with DandiAPIClient() as client:
        paths = []
        for file in client.get_dandiset("000055", "draft").get_assets_with_path_prefix(""):
            paths.append(file.path)
    paths = natsort.natsorted(paths)

    displ_lst, part_lst, time_lst, pose_lst = [], [], [], []
    for pat in range(n_parts):
        fids = [val for val in paths if "sub-" + str(pat + 1).zfill(2) in val]
        for i, fid in enumerate(fids):
            with DandiAPIClient() as client:
                asset = client.get_dandiset("000055", "draft").get_asset_by_path(fid)
                s3_path = asset.get_content_url(follow_redirects=1, strip_query=True)
            f = fs.open(s3_path, "rb")
            file = h5py.File(f)
            with NWBHDF5IO(file=file, mode='r', load_namespaces=True) as io:                
                nwb_file = io.read()

                # Segment data
                events = nwb_file.processing["behavior"].data_interfaces["ReachEvents"]
                times = events.timestamps[:]
                starts = times - before
                stops = times + after

                # Get event hand label
                contra_arm = events.description
                contra_arm = map(lambda x: x.capitalize(), contra_arm.split("_"))
                contra_arm = list(contra_arm)
                contra_arm = "_".join(contra_arm)
                ipsi_arm = (
                    "R" + contra_arm[1:]
                    if contra_arm[0] == "L"
                    else "L" + contra_arm[1:]
                )

                reach_lab = ["contra", "ipsi"]
                for k, reach_arm in enumerate([contra_arm, ipsi_arm]):
                    spatial_series = nwb_file.processing["behavior"].data_interfaces[
                        "Position"
                    ][reach_arm]
                    ep_dat = align_by_times(spatial_series, starts, stops)
                    ep_dat_mag = np.sqrt(
                        np.square(ep_dat[..., 0]) + np.square(ep_dat[..., 1])
                    )

                    # Interpolate and median filter
                    for j in range(ep_dat_mag.shape[0]):
                        df_mag = pd.DataFrame(ep_dat_mag[j, :])
                        df_mag = df_mag.interpolate(method="pad")
                        tmp_val = (
                            df_mag.values.copy().flatten()
                        )  # medfilt(df_mag.values, kernel_size=31)
                        df_mag = pd.DataFrame(tmp_val[::-1])
                        df_mag = df_mag.interpolate(method="pad")
                        ep_dat_mag[j, :] = medfilt(
                            df_mag.values.copy().flatten()[::-1], kernel_size=31
                        )

                    zero_ind = timeseries_time_to_ind(spatial_series, before)
                    base_start_ind = timeseries_time_to_ind(
                        spatial_series, base_start + before
                    )
                    base_end_ind = timeseries_time_to_ind(
                        spatial_series, base_end + before
                    )
                    n_tpoints = ep_dat_mag.shape[1]
                    t_vals = np.arange(n_tpoints) / fs_video - before

                    # Subtract baseline from position data
                    for j in range(ep_dat_mag.shape[0]):
                        curr_magnitude = ep_dat_mag[j, :]
                        curr_magnitude = np.abs(
                            curr_magnitude
                            - np.mean(curr_magnitude[base_start_ind:base_end_ind])
                        )
                        curr_magnitude[np.isnan(curr_magnitude)] = 0
                        displ_lst.extend(curr_magnitude.tolist())
                        part_lst.extend(["P" + str(pat + 1).zfill(2)] * n_tpoints)
                        time_lst.extend(t_vals.tolist())
                        pose_lst.extend([reach_lab[k]] * n_tpoints)

            del nwb_file, io

    df_pose = pd.DataFrame(
        {"Displ": displ_lst, "Sbj": part_lst, "Time": time_lst, "Contra": pose_lst}
    )
    return df_pose, part_lst

def plot_dlc_recon_errs(fig, ax):
    """Plots DeepLabCut reconstruction errors on training and heldout
    images. This information is not present in the NWB files."""
    # DLC reconstruction errors [train set, holdout set]
    sbj_d = {
        "P01": [1.45, 4.27],
        "P02": [1.44, 3.58],
        "P03": [1.58, 6.95],
        "P04": [1.63, 6.02],
        "P05": [1.43, 3.42],
        "P06": [1.43, 6.63],
        "P07": [1.51, 5.45],
        "P08": [1.84, 10.35],
        "P09": [1.4, 4.05],
        "P10": [1.48, 7.59],
        "P11": [1.51, 5.45],
        "P12": [1.52, 4.73],
    }

    train_err = [val[0] for key, val in sbj_d.items()]
    test_err = [val[1] for key, val in sbj_d.items()]

    nsbjs = len(train_err)
    sbj_nums = [val + 1 for val in range(nsbjs)]
    sbj = ["P" + str(val).zfill(2) for val in sbj_nums]

    # Create plot

    ax.bar(sbj, train_err, color="dimgrey")
    ax.bar(sbj, test_err, color="lightgrey")
    ax.bar(sbj, train_err, color="dimgrey")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xticklabels(sbj, rotation=45)
    ax.legend(["Train set", "Holdout set"], frameon=False, fontsize=8)
    ax.tick_params(labelsize=9)
    ax.set_ylabel("Reconstruction error (pixels)")
    ax.set_title("(a) Pose estimation model errors", fontsize=10)