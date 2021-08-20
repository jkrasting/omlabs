import xcompare

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


def parse(cliargs=None, template=False):
    description = """Plot thermocline strength and depth of maximum"""

    if template is True:
        return parser.parse_args(None).__dict__
    else:
        return parser.parse_args(cliargs)


def read(dictArgs):
    dset = xr.open_mfdataset(dictArgs["infile"], use_cftime=True)
    dset_obs = xr.open_dataset(dictArgs["obsfile"], decode_times=False)
    return dset, dset_obs


def _dtdz_max_and_depth(thetao, zcoord_name="z_l"):
    # get the values for the vertical coordinate
    zcoord = thetao[zcoord_name].values

    # forward fill along z-dimension before taking derivative
    thetao = thetao.ffill(dim=zcoord_name)
    dtdz = thetao.fillna(0.0).diff(dim=zcoord_name)

    # take the absolute value of the derivative
    dtdz = np.abs(dtdz)

    # the maximum value of the derivative (i.e. thermocline strength)
    dtdz_max = dtdz.max(dim=zcoord_name)

    # get the z-index of the maximum value and use list comprehension
    # to get the corresponding depth of the max value
    maxind = dtdz.argmax(dim=zcoord_name).values.flatten()
    dtdz_depth = np.array([zcoord[x] for x in maxind]).reshape(dtdz_max.shape)

    # cast the depth of max value back to an DataArray
    dtdz_depth = xr.DataArray(dtdz_depth, coords=dtdz_max.coords)

    # reapply masks
    dtdz_max = xr.where(thetao.isel({zcoord_name: 0}).isnull(), np.nan, dtdz_max)
    dtdz_depth = xr.where(thetao.isel({zcoord_name: 0}).isnull(), np.nan, dtdz_depth)

    # package all fields up in an xr.Dataset() object
    dset_out = xr.Dataset()
    dset_out["dtdz_depth"] = dtdz_depth
    dset_out["dtdz_max"] = dtdz_max

    return dset_out


def calculate(dset, dset_obs):

    # perform a time average
    dset = dset.mean(dim="time").squeeze()
    dset_obs = dset_obs.mean(dim="time").squeeze()

    thetao = dset["thetao"].load()
    dset_out = _dtdz_max_and_depth(thetao)

    ptemp = dset_obs["ptemp"].load()
    dset_obs_out = _dtdz_max_and_depth(ptemp)

    return dset_out, dset_obs_out


def plot(dset_out, dset_obs_out):
    results = xcompare.compare_datasets(dset_out, dset_obs_out)
    figs = [
        xcompare.plot_three_panel(results, "dtdz_max"),
        xcompare.plot_three_panel(results, "dtdz_depth"),
    ]

    return figs


def run():
    # set visual backend
    if dictArgs["interactive"] is False:
        plt.switch_backend("Agg")

    dset, dset_obs = read(dictArgs)
    dset_out, dset_obs_out = calculate(dset, dset_obs)
    figs = plot(dset_out, dset_obs_out)

    filenames = [
        f"{dictArgs['outdir']}/thermocline.dtdz_max",
        f"{dictArgs['outdir']}/thermocline.dtdz_depth",
    ]

    imgbufs = image_handler(figs, dictArgs, filename=filenames)

    return imgbufs


def parse_and_run(cliargs=None):
    """Function to make compatibile with the superwrapper"""
    args = parse(cliargs)
    args = args.__dict__
    imgbuf = run(args)
    return imgbuf


if __name__ == "__main__":
    parse_and_run()
