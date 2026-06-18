import argparse
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml
from joblib import Parallel, delayed  # noqa: F401

# CONFIG_PREPROCESSING = Path("configs/conversion/image_config_2025_6016px.yml")
# CONFIG_INFERENCE = Path("configs/inference/image_config_2025_inference.yml")
N_JOBS = 6
DRY_RUN = False
N_PROJECTS = None
AUTO_PROJECTS = True


def main():
    parser = argparse.ArgumentParser(description="Run image inference on a project.")
    parser.add_argument(
        "--config-preprocessing",
        type=Path,
        help="Path to the preprocessing configuration.",
    )
    parser.add_argument(
        "--config-inference", type=Path, help="Path to the inference configuration."
    )
    args = parser.parse_args()

    CONFIG_PREPROCESSING = args.config_preprocessing
    CONFIG_INFERENCE = args.config_inference

    if AUTO_PROJECTS:
        # read data_dir from CONFIG_PREPROCESSING
        with open(CONFIG_PREPROCESSING, "r") as f:
            config_preprocessing = yaml.safe_load(f)

        with open(CONFIG_INFERENCE, "r") as f:
            config_inference = yaml.safe_load(f)

        data_dir = Path(config_preprocessing["data_dir"])

        model_name = Path(config_inference["model"]).stem
        target_dir = Path(config_inference["output_dir"])  # / model_name

        projects_raw = data_dir.glob("*")
        # filter down to directories starting with "20" (e.g. 20250720-183714_[ - ])
        PROJECTS = [
            p.name for p in projects_raw if p.is_dir() and (p.name.startswith("20"))
        ]
        # filter down to not yet processed projects
        # PROJECTS = [p for p in PROJECTS if not (target_dir / p).exists() else print(f"Project {p} already exists in target directory, skipping...")]

        PROJECTS_RUN = []
        for p in PROJECTS:
            if not (target_dir / p).exists():
                PROJECTS_RUN.append(p)
            else:
                print(f"Project {p} already exists in target directory, skipping...")

        PROJECTS = PROJECTS_RUN
    else:
        raise NotImplementedError(
            "Manual project list is not implemented yet. Please set AUTO_PROJECTS to True."
        )

    # sort projects by name
    PROJECTS.sort()

    [print(p) for p in PROJECTS]

    # run preprocessing and inference
    Parallel(n_jobs=N_JOBS)(
        delayed(run_inference)(project, CONFIG_PREPROCESSING, CONFIG_INFERENCE)
        for project in PROJECTS[:N_PROJECTS]
    )

    # merge final results
    merge_results(target_dir)


def run_inference(project_name: str, config_preprocessing, config_inference):
    """
    Run inference on the project.
    """
    # run preprocessing
    s_preprocessing = f'uv run convert_images.py --config {config_preprocessing} --project-names "{project_name}'
    if not DRY_RUN:
        os.system(s_preprocessing)

    # run inference
    s_inference = f'uv run dbh_inference --config {config_inference} --projects-to-run "{project_name}"'
    if not DRY_RUN:
        os.system(s_inference)


def merge_results(target_dir: Path):
    """
    Merge feature locations from a given directory into a single GeoDataFrame.
    Print the number of features processed.

    Parameters:
        target_dir: Path to the directory containing feature locations

    Output:
        0 on successful completion
    """
    feature_locations = target_dir.rglob("*_feature_locations.gpkg")
    gdf_merged = gpd.GeoDataFrame(
        pd.concat([gpd.read_file(f) for f in feature_locations])
    )

    outfile_base = target_dir / (target_dir.name + "_feature_locations")
    gdf_merged.to_file(f"{outfile_base}.gpkg")
    gdf_merged.to_parquet(f"{outfile_base}.parquet")
    return 0


if __name__ == "__main__":
    # run full project
    main()
