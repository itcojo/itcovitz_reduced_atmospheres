import subprocess


def get_dpi():
    """
    Detects screen size (mm) and provides an appropriate dpi for easy and
    consistent viewing of figures.

    Parameters
    ----------

    Returns
    -------
    dpi : int
        Appropriate dpi for the detected screen.

    """
    # get screen size in units of [mm]
    screens = [l.split()[-3:] for l in subprocess.check_output(
        ["xrandr"]).decode("utf-8").strip().splitlines() if " connected" in l]

    dimensions = []
    for s in screens:
        for item in s:
            if 'mm' in item:
                dimensions.append(item)

    # set appropriate dpi for you screens
    if len(dimensions) == 2:  # single monitor
        # itcovitz X1C7
        if '309mm' in dimensions:
            dpi = 100
        # itcovitz U2415
        elif '518mm' in dimensions:
            dpi = 175

    if len(dimensions) == 4:  # dual monitors
        # itcovitz U2415
        if '518mm' in dimensions:
           dpi = 175

    return dpi
    

