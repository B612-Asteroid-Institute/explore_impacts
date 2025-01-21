import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from adam_core.observations.ades import ADESObservations
from astropy.time import Time
from mpcq import BigQueryMPCClient, MPCObservations
from adam_impact_study.fo_od import _create_fo_tmp_directory
import pathlib
import subprocess
import os

import logging

logger = logging.getLogger(__name__)


FO_BINARY_DIR = pathlib.Path(__file__).parent.parent.parent / "find_orb/find_orb"
LINUX_JPL_PATH = (
    pathlib.Path(__file__).parent.parent.parent
    / "find_orb/.find_orb/linux_p1550p2650.440t"
)

BC405_FILENAME = (
    pathlib.Path(__file__).parent.parent.parent / "find_orb/.find_orb/bc405.dat"
)

def custom_run_fo_od(ades_string: str, output_dir: str):
    fo_tmp_dir = _create_fo_tmp_directory()
    fo_input_file = os.path.join(fo_tmp_dir, "observations.ades")
    with open(fo_input_file, "w") as f:
        f.write(ades_string)

    fo_command = (
        f"{FO_BINARY_DIR}/fo {fo_input_file} -c "
        f"-d 10 "
        f"-D {fo_tmp_dir}/environ.dat "
        f"-O {output_dir}"
    )


    result = subprocess.run(
        fo_command,
        shell=True,
        cwd=fo_tmp_dir,
        text=True,
        capture_output=True,
    )


    _copy_files_from_tmp_to_fo_dir(fo_tmp_dir, fo_result_dir)
    # Remove the tmp directory after copying because it has
    # some large files in it that we don't need

    if result.returncode != 0:
        logger.warning(
            f"Find_Orb failed with return code {result.returncode} for {len(observations)} observations in {fo_result_dir}"
        )
        logger.warning(f"{result.stdout}\n{result.stderr}")
        return Orbits.empty(), ADESObservations.empty(), "Find_Orb failed"

    if not os.path.exists(f"{fo_result_dir}/covar.json") or not os.path.exists(
        f"{fo_result_dir}/total.json"
    ):
        logger.warning("Find_Orb failed, covar.json or total.json file not found")
        return (
            Orbits.empty(),
            ADESObservations.empty(),
            "Find_Orb failed, covar.json or total.json file not found",
        )

    orbit = fo_to_adam_orbit_cov(fo_result_dir)

    # Re-assign orbit_id to the original value
    orbit = orbit.set_column("orbit_id", observations[0].orbit_id)
    rejected = rejected_observations_from_fo(fo_result_dir)

    return orbit, rejected, None


    return result
def lax_ADES_to_string(
    observations: ADESObservations,
    seconds_precision: int = 3,
    columns_precision: dict[str, int] = {
        "ra": 8,
        "dec": 8,
        "rmsRA": 4,
        "rmsDec": 4,
        "mag": 2,
        "rmsMag": 2,
    },
) -> str:
    """
    Write ADES observations to a string.

    Parameters
    ----------
    observations : ADESObservations
        The observations to write to a string.
    obs_contexts : dict[str, ObsContext]
        A dictionary of observatory codes and their corresponding ObsContexts to use
        as the context headers for the different observatory codes in the observations.
    seconds_precision : int, optional
        The precision to use for the seconds in the obsTime field, by default 3.
    columns_precision : dict[str, int], optional
        A dictionary of column names and their corresponding precision to use when writing
        the observations to the file, by default {
            "ra": 8,
            "dec": 8,
            "rmsRA" : 4,
            "rmsDec" : 4,
            "mag": 2,
            "rmsMag": 2,
        }
        The MPC enforces strict limits on these and submitters may need permission to send
        high-precision data.

    Returns
    -------
    ades_string : str
        The ADES observations as a string.
    """
    ades_string = "# version=2022\n"

    unique_observatories = observations.stn.unique().to_numpy(zero_copy_only=False)
    unique_observatories.sort()

    for obs in unique_observatories:

        observations_obscode = observations.select("stn", obs)
        observations_obscode = observations_obscode.sort_by(
            [
                ("provID", "ascending"),
                ("permID", "ascending"),
                ("trkSub", "ascending"),
                ("obsTime.days", "ascending"),
                ("obsTime.nanos", "ascending"),
            ]
        )

        id_present = False
        if not pc.all(pc.is_null(observations_obscode.permID)).as_py():
            id_present = True
        if not pc.all(pc.is_null(observations_obscode.provID)).as_py():
            id_present = True
        if not pc.all(pc.is_null(observations_obscode.trkSub)).as_py():
            id_present = True

        if not id_present:
            err = (
                "At least one of permID, provID, or trkSub should\n"
                "be present in observations."
            )
            raise ValueError(err)


        # Write the observations block (we first convert
        # to a pandas dataframe)
        ades = observations_obscode.to_dataframe()

        # Convert the timestamp to ISOT with the desired precision
        observation_times = Time(
            observations_obscode.obsTime.rescale("utc")
            .mjd()
            .to_numpy(zero_copy_only=False),
            format="mjd",
            precision=seconds_precision,
        )
        ades.insert(
            4,
            "obsTime",
            np.array([i + "Z" for i in observation_times.utc.isot]),
        )
        ades.drop(columns=["obsTime.days", "obsTime.nanos"], inplace=True)

        # Multiply rmsRA by cos(dec) since ADES wants the random component in rmsRAcosDec
        ades.loc[:, "rmsRA"] *= np.cos(np.radians(ades["dec"]))

        # Convert rmsRA and rmsDec to arcseconds
        ades.loc[:, "rmsRA"] *= 3600
        ades.loc[:, "rmsDec"] *= 3600

        ades.dropna(how="all", axis=1, inplace=True)

        # Change the precision of some of the columns to conform
        # to MPC standards
        for col, prec_col in columns_precision.items():
            if col in ades.columns:
                ades[col] = [
                    f"{i:.{prec_col}f}" if i is not None or not np.isnan(i) else ""
                    for i in ades[col]
                ]

        ades_string += ades.to_csv(
            sep="|", header=True, index=False, float_format="%.16f"
        )

    return ades_string



# Collect observations from MPC
def collect_ades_observations(object_id: str) -> str:
    client = BigQueryMPCClient()
    observations: MPCObservations = client.query_observations([object_id])

    # Cast the MPCObservations to ADESObservations
    ades_observations = ADESObservations.from_kwargs(
        permID=observations.permid,
        provID=observations.provid,
        trkSub=observations.trksub,
        obsSubId=observations.obsid,
        obsTime=observations.obstime,
        ra=observations.ra,
        dec=observations.dec,
        rmsRA=observations.rmsra,
        rmsDec=observations.rmsdec,
        mag=observations.mag,
        rmsMag=observations.rmsmag,
        band=observations.band,
        stn=observations.stn,
        mode=pa.repeat("CCD", len(observations)),
        astCat=pa.repeat("N/A", len(observations)),
    )

    ades_string = lax_ADES_to_string(ades_observations)
    return ades_string


def 





# Send ADES to fo to fit orbit
# generate variants and calculate impacts




def run():
    pass

if __name__ == "__main__":
    run()