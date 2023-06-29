# # streaming
# import fsspec
# from fsspec.implementations.caching import CachingFileSystem

# fs = CachingFileSystem(
#     fs=fsspec.filesystem("http")
# )


# DANDI/NWB
from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient

# Numerical
import natsort
import fsspec
import h5py
import numpy as np

# general
from tqdm import tqdm

# local
from plot_utils.elecs import identify_elecs

def load_data_characteristics(nparts=12, fs=None):
    """Load data characteristics including the number of
    good and total ECoG electrodes, hemisphere implanted,
    and number of recording days for each participant."""
    with DandiAPIClient() as client:
        paths = []
        for file in client.get_dandiset("000055", "draft").get_assets_with_path_prefix(""):
            paths.append(file.path)
    paths = natsort.natsorted(paths)

    n_elecs_tot, n_elecs_good = [], []
    rec_days, hemis, n_elecs_surf_tot, n_elecs_depth_tot = [], [], [], []
    n_elecs_surf_good, n_elecs_depth_good = [], []
    for part_ind in tqdm(range(nparts)):
        fids = [val for val in paths if "sub-" + str(part_ind + 1).zfill(2) in val]
        rec_days.append(len(fids))
        for fid in fids[:1]:
            with DandiAPIClient() as client:
                asset = client.get_dandiset("000055", "draft").get_asset_by_path(fid)
                s3_path = asset.get_content_url(follow_redirects=1, strip_query=True)
            f = fs.open(s3_path, "rb")
            file = h5py.File(f)
            with NWBHDF5IO(file=file, mode='r', load_namespaces=True) as io:
                nwb = io.read()

                # Determine good/total electrodes
                n_elecs_good.append(np.sum(nwb.electrodes["good"][:]))
                n_elecs_tot.append(len(nwb.electrodes["good"][:]))

                # Determine implanted hemisphere
                c_wrist = (
                    nwb.processing["behavior"].data_interfaces["ReachEvents"].description[0]
                )
                hemis.append("L" if c_wrist == "r" else "R")

                # Determine surface vs. depth electrode count
                is_surf = identify_elecs(nwb.electrodes["group_name"][:])
                n_elecs_surf_tot.append(np.sum(is_surf))
                n_elecs_depth_tot.append(np.sum(1 - is_surf))
                n_elecs_surf_good.append(
                    np.sum(nwb.electrodes["good"][is_surf.nonzero()[0]])
                )
                n_elecs_depth_good.append(
                    np.sum(nwb.electrodes["good"][(1 - is_surf).nonzero()[0]])
                )

            del nwb, io

    part_nums = [val + 1 for val in range(nparts)]
    part_ids = ["P" + str(val).zfill(2) for val in part_nums]

    return [
        rec_days,
        hemis,
        n_elecs_surf_tot,
        n_elecs_surf_good,
        n_elecs_depth_tot,
        n_elecs_depth_good,
        part_nums,
        part_ids,
        n_elecs_good,
        n_elecs_tot,
    ]