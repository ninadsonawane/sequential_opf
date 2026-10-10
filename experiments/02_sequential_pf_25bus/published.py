"""Transcription of P3 Tables4--7. These are targets, never fitted inputs."""

import numpy as np

VM = np.array(
    [
        1.010,
        1.000,
        1.011,
        1.031,
        1.006,
        1.000,
        0.990,
        1.020,
        1.005,
        1.007,
        0.980,
        1.021,
        1.020,
        1.013,
        1.003,
        0.993,
        0.985,
        0.991,
        1.003,
        1,
        1,
        1.023,
        1.010,
        1,
        0.999,
    ]
)
ANGLE = np.array(
    [
        0,
        -1.73,
        -1.69,
        -3.05,
        -4.35,
        0,
        -0.04,
        -0.06,
        -0.19,
        -0.14,
        0.37,
        -0.39,
        0.06,
        -0.51,
        -0.39,
        -0.19,
        0.59,
        0.39,
        -0.27,
        2.35,
        0,
        1.43,
    ]
)
FLOWS = [
    ("HVAC", 4, 10, -33.97, -559.60, 34.06, -567.94),
    ("LFAC", 20, 21, 156.58, -120.79, -155.21, -54.09),
    ("LFAC", 20, 22, 43.42, -192.49, -42.76, 10),
    ("DC", 23, 24, 96.10, None, -95.22, None),
    ("DC", 23, 25, 99.68, None, -98.73, None),
    ("DC", 24, 25, 3.548, None, -3.546, None),
]
CONVERTERS = dict(
    ps_mw=[-200, 89.58, 100, 148.37, 40, -155.21, -42.76],
    qs_mvar=[-30, -15.99, 5, -30, -10, -54.09, 10],
    vm_pu=[1.01, 0.985, 0.991, 1.011, 1.003, 1, 1.023],
    angle_deg=[-4.35, 0.596, 0.397, -1.688, -0.389, 0, 1.432],
    pc_mw=[-199.34, 89.66, 100.16, 148.73, 40.02, -154.78, -42.72],
    qc_mvar=[72.51, -0.50, 23.88, 23.49, -13.73, 13.33, 5.37],
    vc_pu=[1.058, 0.961, 1.030, 1.002, 0.9681, 0.937, 1.041],
    converter_angle_deg=[-35.38, 15.69, 15.99, 22.07, 6.117, -27.02, -4.907],
    switching_loss_mw=[3.56, 2.01, 2.12, 3.17, 1.38, 2.88, 1.32],
)

AUDIT = [
    dict(
        item="Stagg5 topology/base loads",
        status="verified source",
        detail="Stagg/El-Abiad 5-bus, 100MVA, 345kV; official MatACDC case5_stagg.m. Not MATPOWER case5 PJM.",
    ),
    dict(
        item="IEEE14 topology/base loads",
        status="standard source",
        detail="Public standard IEEE14 via PYPOWER, whose case data originated from MATPOWER. Illinois reference links the same test system; authors final modified data not available.",
    ),
    dict(
        item="Final load and generator vectors",
        status="missing",
        detail="P3 says load approximately five times initial. Reconstruction chooses exactly 5.0, retains original non-slack Pg, and does not fit reported voltages.",
    ),
    dict(
        item="345kV internal branch RLC/taps",
        status="missing",
        detail="P3 specifies EPRI conductor/tower selection, 50km internal branches, but publishes only one Appendix geometry. Reconstruction extends the Appendix 500kV RLC to all 345kV internal branches, unity taps, no fixed shunts.",
    ),
    dict(
        item="LFAC branch data",
        status="strongly supported reconstruction",
        detail="Two 1000km 500kV lines; 10Hz inferred from Appendix R10 and reported charging flows. Nominal-pi model selected. Frequency/model not explicitly stated for the case.",
    ),
    dict(
        item="DC conductor/base/poles",
        status="missing",
        detail="320kV, 1000km triangle known. Conductors and pole configuration not specified. Reconstruction uses R=.0265/2 ohm/km and single effective-pole DC equation; no calibration against Table5.",
    ),
    dict(
        item="Converter Eq9",
        status="notation conflict",
        detail="Printed Ic=Is+Vf/Zt uses transformer impedance in filter current. Fig4 KCL requires Ic=Is+Yf Vf.",
    ),
    dict(
        item="Converter Table3 filter",
        status="notation conflict",
        detail="Printed Zf=j.0887 conflicts with capacitive filter and reported results. Implement Yf=+j.0887 (equivalently Zf=-j/.0887), consistent with Fig4 and MatACDC convention.",
    ),
    dict(
        item="DC voltage at bus23",
        status="table conflict",
        detail="Table4 says 1.010 pu; Table6 says 1.001 pu for the same DC bus.",
    ),
    dict(
        item="AC voltage at VSC5/PCC16",
        status="table conflict",
        detail="Table4 says .993 pu / -.19 deg; Table7 says 1.003 pu / -.389 deg. The Table7 values coincide with Table4 bus15, despite Table2 and Fig6 specifying bus16.",
    ),
    dict(
        item="AC voltage at VSC1/PCC5",
        status="rounding/conflict",
        detail="Table4 gives 1.006 pu; Table6 gives 1.01 pu. Converter component checks evaluate both, without using them to fit the network.",
    ),
    dict(
        item="Converter capability/rating bases",
        status="ambiguous",
        detail="The stated 1.05pu current bound is exceeded by the Table6/7 schedules under the common 100MVA interpretation that reproduces losses. Rating/base conversion or limit-switching details required. Flag violations; no claim of capability-feasible reproduction.",
    ),
    dict(
        item="Anderson settings and stopping",
        status="partly missing",
        detail="mmax=8 known; gammaMax and exact 25bus tolerance unknown. Use gammaMax1000, 1e-9 inner mismatch, 1e-8 outer DC interface mismatch. Preserve bus order; no fastmath.",
    ),
    dict(
        item="Data availability",
        status="on request",
        detail="Paper states data will be made available on request. No author data/code is bundled or publicly linked in the PDF; no message sent to authors.",
    ),
]
