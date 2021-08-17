from .thermocline import parse, read, calculate, plot, run, parse_and_run

__description__ = "Plots thermocline strength and depth relative to WOA"
__ppstreams__ = [
    "ocean_month_z_d2_refined/av",
    "ocean_monthly_z_d2/ts",
    "ocean_annual_z/av",
]
__ppvars__ = ["thetao"]
