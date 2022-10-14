"""Generate training data."""
import argparse
import logging
import os
import random
import uuid
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyfstat
from scipy import stats
from tqdm.auto import tqdm

pyfstat.logger.setLevel(logging.ERROR)


def main() -> None:
    """Generate training data."""
    parser = argparse.ArgumentParser("Training data generator.")
    parser.add_argument("save_path", type=Path)
    parser.add_argument("--num_signals", type=int, default=10000)

    args = parser.parse_args()

    save_path = args.save_path / "train"
    save_path.mkdir(exist_ok=True, parents=True)

    snrs = np.zeros(args.num_signals)

    metadata = []
    for i in tqdm(range(args.num_signals)):

        # These parameters describe background noise and data format
        writer_kwargs = {
            "tstart": 1238166018,
            "duration": 4 * 30 * 86400,
            "detectors": "H1,L1",
            "sqrtSX": 1e-23,
            "Tsft": 1800,
            "SFTWindowType": "tukey",
            "SFTWindowBeta": 0.01,
            "Band": 0.2,
        }
        # This class allows us to sample signal parameters from a specific population.
        # Implicitly, sky positions are drawn uniformly across the celestial sphere.
        # PyFstat also implements a convenient set of priors to sample a population
        # of isotropically oriented neutron stars.
        signal_parameters_generator = pyfstat.AllSkyInjectionParametersGenerator(
            priors={
                "tref": writer_kwargs["tstart"],
                "F0": {"uniform": {"low": 50.0, "high": 500.0}},
                "F1": {"uniform": {"low": -1e-23, "high": 1e-23}},
                "F2": 0,
                "h0": lambda: writer_kwargs["sqrtSX"] / stats.uniform(1, 100).rvs(),
                # pyfstat.injection_parameters.isotropic_amplitude_priors
                "cosi": {"uniform": {"low": -1.0, "high": 1.0}},
                "psi": {
                    "uniform": {"low": -0.7853981633974483, "high": 0.7853981633974483}
                },
                "phi": {"uniform": {"low": 0, "high": 6.283185307179586}},
            },
        )

        target = random.choice([0.0, 1.0])
        label = uuid.uuid4().hex[:10]

        # Draw signal parameters.
        # Noise can be drawn by setting `params["h0"] = 0
        params = signal_parameters_generator.draw()

        if target == 0.0:
            params["h0"] = 0.0

        writer_kwargs["outdir"] = save_path.parent / "raw" / label
        writer_kwargs["label"] = label

        writer = pyfstat.Writer(**writer_kwargs, **params)

        try:
            writer.make_data()
        except Exception as e:
            print(e)
            continue

        # SNR can be compute from a set of SFTs for a specific set
        # of parameters as follows:
        snr = pyfstat.SignalToNoiseRatio.from_sfts(
            F0=writer.F0, sftfilepath=writer.sftfilepath
        )
        squared_snr = snr.compute_snr2(
            Alpha=writer.Alpha,
            Delta=writer.Delta,
            psi=writer.psi,
            phi=writer.phi,
            h0=writer.h0,
            cosi=writer.cosi,
        )
        snrs[i] = np.sqrt(squared_snr)

        # Data can be read as a numpy array using PyFstat
        frequency, timestamps, amplitudes = pyfstat.utils.get_sft_as_arrays(
            writer.sftfilepath
        )
        with h5py.File(save_path / f"{label}.hdf5", mode="w") as file:
            file_id = file.create_group(label)
            file_id.create_dataset("frequency_Hz", data=frequency)

            h1_amplitudes = amplitudes["H1"][:-1, :]
            h1 = file_id.create_group("H1")
            h1.create_dataset("SFTs", data=h1_amplitudes)
            h1.create_dataset("timestamps_GPS", data=timestamps["H1"])

            l1_amplitudes = amplitudes["L1"][:-1, :]
            l1 = file_id.create_group("L1")
            l1.create_dataset("SFTs", data=l1_amplitudes)
            l1.create_dataset("timestamps_GPS", data=timestamps["L1"])

        metadata.append({"id": label, **params, **writer_kwargs, "target": target})

        file_paths = writer.sftfilepath.split(";")
        for file_path in file_paths:
            os.remove(file_path)

    df = pd.DataFrame.from_records(metadata)
    df.to_csv(save_path.parent / "train_labels.csv", index=False)


if __name__ == "__main__":
    main()
