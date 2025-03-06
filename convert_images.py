import os
from pathlib import Path

import numpy as np
import typer
from tqdm import tqdm
from typer_config import use_yaml_config
from typer_config.decorators import use_yaml_config
from typing_extensions import Annotated

app = typer.Typer()


def process_files(
    indir: Path,
    mipps_dir: Path,
    chunksize: int,
    outdir: Path,
    mipps_bin: str,
    mipps_file: str,
):
    infiles = list(indir.glob("*RGB*/*"))
    infiles = [f'"{str(f)}"' for f in infiles]  # get posix files

    # setup mipps call
    os.chdir(mipps_dir)

    split = len(infiles) // chunksize
    if split == 0:
        split += 1
    for df in tqdm(np.array_split(infiles, split)):
        outlist = " ".join(df)
        os.makedirs(outdir, exist_ok=True)
        s = f'{mipps_bin} -c={mipps_file} -o="{outdir}" -j=4 {outlist}'
        # run mipps call
        os.system(s)


@app.command()
@use_yaml_config()
def main(
    data_dir: Annotated[Path, typer.Option(help="Path to the data directory")] = Path(
        r"N:\Response\Restricted_Airborne\MACS\Alaska\2024_Perma-X_Alaska\01_raw_data"
    ),
    out_dir_base: Annotated[
        Path, typer.Option(help="Path to the output directory")
    ] = Path(r"S:\p_initze\yolov8_object_detection\data"),
    chunksize: Annotated[int, typer.Option(help="Chunk size for processing")] = 20,
    mipps_file: Annotated[
        str, typer.Option(help="Path to the MIPPS file")
    ] = r"N:\Response\Restricted_Airborne\MACS\Alaska\2024_Perma-X_Alaska\05_mipps_scripts\MACS_Polar_RGB_mipps\111498_per_pixel_scale_jpg75_rescale.mipps",
    mipps_dir: Annotated[
        str, typer.Option(help="Path to the MIPPS directory")
    ] = r"C:\Program Files\DLR MACS-Box\bin",
    mipps_bin: Annotated[
        str, typer.Option(help="Path to the MIPPS binary")
    ] = r"..\tools\Conv\mipps.exe",
    n_projects: Annotated[
        int,
        typer.Option(help="Process defined number of projects (e.g. for debugging)"),
    ] = None,
    filter: Annotated[
        str, typer.Option(help="Filter to boil down to specific project name")
    ] = None,
):
    """
    Process MACS images and convert them to jpg files.
    """
    # List all subdirectories
    projects = [
        x for x in data_dir.iterdir() if (x.is_dir() and x.name.startswith("202"))
    ]

    # filter to selected regex
    if filter:
        projects = [x for x in projects if filter in x.name]

    # Process projects
    for project_name in projects[:n_projects]:
        print(f"Processing: {project_name}")
        indir = data_dir / project_name
        outdir = out_dir_base / indir.name
        try:
            if not outdir.exists():
                process_files(
                    indir, Path(mipps_dir), chunksize, outdir, mipps_bin, mipps_file
                )
            else:
                print("Outdir exists, skip processing!")
        except Exception as e:
            print(f"Error processing {project_name}: {str(e)}")
            continue


if __name__ == "__main__":
    app()
