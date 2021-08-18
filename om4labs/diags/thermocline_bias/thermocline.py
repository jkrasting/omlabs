import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


def parse():
    return


def read(dictArgs):
    dset = xr.open_mfdataset(dictArgs["infile"], use_cftime=True)
    return dset


def calculate(dset):

    # set default names for variables and coordinates
    varname = "thetao"
    zcoord_name = "z_l"

    # perform a time average
    dset_out = dset.mean(dim="time")
    thetao = dset_out[varname].load()
    zcoord = dset[zcoord_name].values

    # forward fill along z-dimension before taking derivative
    thetao = thetao.ffill(dim="z_l")
    dtdz = thetao.fillna(0.0).diff(dim="z_l")

    # take the absolute value of the derivative
    dtdz = np.abs(dtdz)

    # the maximum value of the derivative (i.e. thermocline strength)
    dtdz_max = dtdz.max(dim="z_l")

    # get the z-index of the maximum value and use list comprehension
    # to get the corresponding depth of the max value
    maxind = dtdz.argmax(dim="z_l").values.flatten()
    dtdz_depth = np.array([zcoord[x] for x in maxind]).reshape(dtdz_max.shape)

    # cast the depth of max value back to an DataArray and mask it
    dtdz_depth = xr.DataArray(dtdz_depth, coords=dtdz_max.coords)
    dtdz_depth = xr.where(thetao.isel({zcoord_name: 0}).isnull(), np.nan, dtdz_depth)

    # package all fields up in an xr.Dataset() object
    dset_out = xr.Dataset()
    dset_out["dtdz_depth"] = dtdz_depth
    dset_out["dtdz_max"] = dtdz_max

    return dset_out


def plot(dset_out):
    plotarr = dset_out["dtdz_depth"]
    fig = plt.figure()
    plt.pcolormesh(plotarr, vmin=0.0, vmax=1000.0)
    plt.colorbar()

    return fig


def run():
    dset = read(dictArgs)
    dset_out = compute(dset)
    fig = plot(dset_out)

    return


def parse_and_run():
    return
