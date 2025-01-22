import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from adam_core.observations.ades import ADESObservations
from astropy.time import Time
from mpcq import BigQueryMPCClient, MPCObservations
from adam_impact_study.fo_od import _create_fo_tmp_directory, _copy_files_from_tmp_to_fo_dir, fo_to_adam_orbit_cov, rejected_observations_from_fo
from adam_core.orbits import Orbits, VariantOrbits
from adam_core.dynamics.impacts import EarthImpacts
from adam_core.dynamics.impacts import calculate_impact_probabilities
from adam_assist import ASSISTPropagator
import pathlib
import subprocess
import os
import shutil
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

def custom_run_fo_od(ades_string: str, output_dir: str, fo_binary_dir: str = FO_BINARY_DIR, bc405_filename: str = BC405_FILENAME, linus_jpl_path: str = LINUX_JPL_PATH):
    print("Fo binary dir: ", FO_BINARY_DIR)
    #delete and recreate output dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    fo_tmp_dir = _create_fo_tmp_directory(
        fo_binary_dir=FO_BINARY_DIR,
        bc405_filename=BC405_FILENAME,
        linus_jpl_path=LINUX_JPL_PATH
    )
    fo_input_file = os.path.join(fo_tmp_dir, "observations.ades")
    with open(fo_input_file, "w") as f:
        f.write(ades_string)

    fo_command = (
        f"{FO_BINARY_DIR}/fo {fo_input_file} -c "
        f"-d 10 "
        f"-D {fo_tmp_dir}/environ.dat "
    )

    result = subprocess.run(
        fo_command,
        shell=True,
        cwd=fo_tmp_dir,
        text=True,
        capture_output=True,
    )

    _copy_files_from_tmp_to_fo_dir(fo_tmp_dir, output_dir)

    if result.returncode != 0:
        logger.warning(
            f"Find_Orb failed with return code {result.returncode} in {fo_tmp_dir}"
        )
        logger.warning(f"{result.stdout}\n{result.stderr}")
        return Orbits.empty(), ADESObservations.empty(), "Find_Orb failed"

    if not os.path.exists(f"{output_dir}/covar.json") or not os.path.exists(
        f"{output_dir}/total.json"
    ):
        logger.warning("Find_Orb failed, covar.json or total.json file not found")
        return (
            Orbits.empty(),
            ADESObservations.empty(),
            "Find_Orb failed, covar.json or total.json file not found",
        )

    orbit = fo_to_adam_orbit_cov(output_dir)

    rejected = rejected_observations_from_fo(output_dir)

    return orbit, rejected, None


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

    observations = observations.sort_by(
        [
            ("provID", "ascending"),
            ("permID", "ascending"),
            ("trkSub", "ascending"),
            ("obsTime.days", "ascending"),
            ("obsTime.nanos", "ascending"),
        ]
    )

    id_present = False
    if not pc.all(pc.is_null(observations.permID)).as_py():
        id_present = True
    if not pc.all(pc.is_null(observations.provID)).as_py():
        id_present = True
    if not pc.all(pc.is_null(observations.trkSub)).as_py():
        id_present = True

    if not id_present:
        err = (
            "At least one of permID, provID, or trkSub should\n"
            "be present in observations."
        )
        raise ValueError(err)


    # Write the observations block (we first convert
    # to a pandas dataframe)
    ades = observations.to_dataframe()

    # Convert the timestamp to ISOT with the desired precision
    observation_times = Time(
        observations.obsTime.rescale("utc")
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
    #ades.loc[:, "rmsRA"] *= 3600
    # ades.loc[:, "rmsDec"] *= 3600

    ades.dropna(how="all", axis=1, inplace=True)

    # Change the precision of some of the columns to conform
    # to MPC standards
    for col, prec_col in columns_precision.items():
        if col in ades.columns:
            ades[col] = [
                f"{i:.{prec_col}f}" if i is not None or not np.isnan(i) else ""
                for i in ades[col]
            ]

    # Replace NaN values with empty strings more explicitly
    ades = ades.replace({np.nan: "", None: "", "nan": ""})
    ades = ades.fillna("")  # Catch any remaining NaN values

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
        obsSubID=observations.obssubid,
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


def run_custom_monte_carlo(orbit: Orbits, n_variants: int = 1000, num_days: int = 30, max_processes: int = 1, seed: int = 42) -> EarthImpacts:
    #generate variants
        variants = VariantOrbits.create(
            orbit, method="monte-carlo", num_samples=n_variants, seed=seed
        )

        propagator = ASSISTPropagator(min_dt=1e-9, initial_dt=1e-6, epsilon=1e-6, adaptive_mode=1)

        final_orbit_states, impacts = propagator.detect_impacts(
            variants,
            num_days,
            max_processes=max_processes,
        )

        impacts.to_parquet(f"impacts.parquet")
        final_orbit_states.to_parquet(f"final_orbit_states.parquet")
        ip = calculate_impact_probabilities(final_orbit_states, impacts)

        print(f"Impact probabilities: {ip.cumulative_probability}")

        return ip.to_dataframe()




def run():
    pass

if __name__ == "__main__":

    object_id = "2024 YR4"
    ades_string = collect_ades_observations(object_id)
    orbit, rejected, error = custom_run_fo_od(ades_string, "fo_result")
    ip = run_custom_monte_carlo(orbit, num_days=3650, n_variants=10000, max_processes=8)
    print(ip)