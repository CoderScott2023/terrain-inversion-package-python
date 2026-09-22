"""
LUNAR SOUTH POLE DATASET BUILDER
================================

Downloads / extracts:

1. LROC NAC South Pole panchromatic imagery
   ~1 m/pixel

2. LOLA global DEM
   ~118 m/pixel at the equator
   Only the requested bbox is extracted.

Output:
    lunar_south_pole/
        clipped/
            lroc_nac_panchromatic.tif
            lola_118m_dem.tif
        bbox.geojson
        raw/

IMPORTANT
---------
The LROC polar mosaic is enormous.

START WITH A SMALL BBOX.

Example test:
    lat: -86.0 to -85.9
    lon: 10 to 15

Once that works, expand it.

Dependencies:
    pip install requests rasterio numpy geopandas shapely tqdm
"""

import os
import re
import math
import requests
import numpy as np

from tqdm import tqdm

import rasterio
from rasterio.windows import from_bounds
from rasterio.merge import merge
from rasterio.warp import (
    transform_bounds,
    reproject,
    Resampling,
)

import geopandas as gpd
from shapely.geometry import Polygon


# ============================================================
# USER CONFIGURATION
# ============================================================

OUTPUT_DIR = "./lunar_south_pole"

# ------------------------------------------------------------
# YOUR LUNAR BBOX
# ------------------------------------------------------------
#
# IMPORTANT:
# This is longitude/latitude in degrees.
#
# Start SMALL:
#

MIN_LAT = -86.0
MAX_LAT = -85.9

MIN_LON = 10.0
MAX_LON = 15.0

#
# For the whole 85.5-86 S ring you would use:
#
# MIN_LAT = -86.0
# MAX_LAT = -85.5
# MIN_LON = -180
# MAX_LON = 180
#
# DO NOT START WITH THAT.
#


# ------------------------------------------------------------
# LROC
# ------------------------------------------------------------

LROC_BASE = (
    "https://pds.lroc.asu.edu/data/"
    "LRO-L-LROC-5-RDR-V1.0/"
    "LROLRC_2001/"
    "EXTRAS/BROWSE/NAC_POLE/NAC_POLE_SOUTH/"
)


# ------------------------------------------------------------
# LOLA
# ------------------------------------------------------------

LOLA_URL = (
    "https://planetarymaps.usgs.gov/mosaic/"
    "Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"
)


# ------------------------------------------------------------
# REQUEST SETTINGS
# ------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


# ============================================================
# DIRECTORIES
# ============================================================

RAW_DIR = os.path.join(
    OUTPUT_DIR,
    "raw",
)

CLIPPED_DIR = os.path.join(
    OUTPUT_DIR,
    "clipped",
)

os.makedirs(
    RAW_DIR,
    exist_ok=True,
)

os.makedirs(
    CLIPPED_DIR,
    exist_ok=True,
)


# ============================================================
# UTILITY
# ============================================================

def print_bbox():

    print()
    print("=" * 70)
    print("REQUESTED LUNAR BBOX")
    print("=" * 70)

    print(
        f"Latitude : {MIN_LAT} -> {MAX_LAT}"
    )

    print(
        f"Longitude: {MIN_LON} -> {MAX_LON}"
    )

    print()


# ============================================================
# DOWNLOAD FILE
# ============================================================

def download_file(
    url,
    output_path,
):

    if os.path.exists(output_path):

        size_gb = (
            os.path.getsize(output_path)
            / 1024**3
        )

        print()
        print(
            f"Already exists: {output_path}"
        )

        print(
            f"Size: {size_gb:.2f} GB"
        )

        return

    print()
    print("=" * 70)
    print("DOWNLOADING")
    print("=" * 70)

    print(url)
    print()

    print(
        f"Destination:\n{output_path}"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        stream=True,
        timeout=120,
    )

    response.raise_for_status()

    total = int(
        response.headers.get(
            "content-length",
            0,
        )
    )

    with open(
        output_path,
        "wb",
    ) as f:

        if total:

            with tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=os.path.basename(
                    output_path
                ),
            ) as pbar:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:

                        f.write(chunk)

                        pbar.update(
                            len(chunk)
                        )

        else:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    f.write(chunk)

    print()
    print("Download complete.")


# ============================================================
# LROC TILE MATH
# ============================================================

def normalize_lon(lon):

    while lon > 180:
        lon -= 360

    while lon < -180:
        lon += 360

    return lon


def longitude_difference(a, b):

    d = abs(a - b)

    if d > 180:
        d = 360 - d

    return d


def tile_filename(
    center_lat,
    center_lon,
):

    """
    Construct actual LROC filename.

    Example:

        center_lat = -86.0
        center_lon = 11.2

    becomes:

        NAC_POLE_P860S0112.TIF
    """

    lat_code = int(
        round(
            abs(center_lat) * 10
        )
    )

    lon_code = int(
        round(
            normalize_lon(center_lon)
            % 360
            * 10
        )
    )

    return (
        f"NAC_POLE_P{lat_code:03d}"
        f"S{lon_code:04d}.TIF"
    )


# ============================================================
# DETERMINE LROC TILES
# ============================================================

def get_candidate_lroc_tiles():

    """
    The outer south-pole NAC mosaic is divided into
    approximately 22.5-degree longitude sectors.

    For the 85.5-86.5 S band, the nominal centers are:

        11.25
        33.75
        56.25
        ...
        348.75

    The P860 tiles therefore have names such as:

        NAC_POLE_P860S0112.TIF
        NAC_POLE_P860S0337.TIF
        NAC_POLE_P860S0562.TIF

    """

    print()
    print("=" * 70)
    print("DETERMINING LROC TILES")
    print("=" * 70)

    # --------------------------------------------------------
    # South-pole mosaic outer band
    # --------------------------------------------------------

    center_lat = -86.0

    # 16 sectors around 360 degrees
    centers = []

    for i in range(16):

        lon = (
            11.25
            + i * 22.5
        )

        centers.append(
            normalize_lon(lon)
        )

    selected = []

    for lon in centers:

        # ----------------------------------------------------
        # Latitude check
        #
        # P860 covers approximately the -86 region.
        # We allow a generous 0.6 degree margin.
        # ----------------------------------------------------

        if (
            MAX_LAT < -86.6
            or MIN_LAT > -85.4
        ):
            continue

        # ----------------------------------------------------
        # Longitude
        #
        # Each tile is roughly 22.5 degrees wide.
        # ----------------------------------------------------

        if (
            MIN_LON <= -180
            and MAX_LON >= 180
        ):

            intersects = True

        else:

            # Handle ordinary bbox
            if MIN_LON <= MAX_LON:

                intersects = (
                    longitude_difference(
                        lon,
                        (MIN_LON + MAX_LON) / 2,
                    )
                    <= (
                        (MAX_LON - MIN_LON)
                        / 2
                        + 11.25
                    )
                )

            else:

                # Dateline-crossing bbox
                intersects = (
                    lon >= MIN_LON
                    or lon <= MAX_LON
                    or longitude_difference(
                        lon,
                        180
                    ) <= 11.25
                )

        if intersects:

            filename = tile_filename(
                center_lat,
                lon,
            )

            selected.append(
                (
                    filename,
                    lon,
                )
            )

    print()
    print(
        f"Selected {len(selected)} "
        f"LROC tile(s):"
    )

    for filename, lon in selected:

        print(
            f"  {filename}"
            f"    center longitude ≈ {lon:.2f}°"
        )

    return [
        filename
        for filename, _ in selected
    ]


# ============================================================
# CHECK LROC URL
# ============================================================

def verify_url(url):

    """
    Test whether the file exists before starting a
    multi-gigabyte download.
    """

    print()
    print("Checking LROC URL:")
    print(url)

    try:

        r = requests.head(
            url,
            headers=HEADERS,
            allow_redirects=True,
            timeout=60,
        )

        print(
            f"HTTP status: {r.status_code}"
        )

        if r.status_code == 200:
            return True

        # Some NASA servers don't handle HEAD correctly.
        if r.status_code in (
            403,
            405,
        ):

            print(
                "HEAD request not supported; "
                "trying GET metadata request."
            )

            r = requests.get(
                url,
                headers={
                    **HEADERS,
                    "Range": "bytes=0-1023",
                },
                stream=True,
                timeout=60,
            )

            print(
                f"GET status: {r.status_code}"
            )

            return r.status_code in (
                200,
                206,
            )

        return False

    except Exception as e:

        print(
            f"URL verification failed: {e}"
        )

        return False


# ============================================================
# DOWNLOAD LROC
# ============================================================

def download_lroc():

    tiles = get_candidate_lroc_tiles()

    if not tiles:

        raise RuntimeError(
            "No LROC tiles intersect the requested bbox."
        )

    paths = []

    print()
    print("=" * 70)
    print("LROC DOWNLOAD")
    print("=" * 70)

    for filename in tiles:

        url = (
            LROC_BASE
            + filename
        )

        local_path = os.path.join(
            RAW_DIR,
            filename,
        )

        # ----------------------------------------------------
        # Check URL first
        # ----------------------------------------------------

        if not verify_url(url):

            raise RuntimeError(
                "\n"
                "LROC file was not found:\n"
                f"{url}\n\n"
                "This means the tile naming/path "
                "has changed on the archive."
            )

        # ----------------------------------------------------
        # Download
        # ----------------------------------------------------

        download_file(
            url,
            local_path,
        )

        paths.append(
            local_path
        )

    return paths


# ============================================================
# MERGE LROC TILES
# ============================================================

def merge_lroc_tiles(paths):

    if len(paths) == 1:

        print()
        print(
            "Only one LROC tile needed; "
            "skipping merge."
        )

        return paths[0]

    print()
    print("=" * 70)
    print("MERGING LROC TILES")
    print("=" * 70)

    datasets = []

    try:

        for path in paths:

            print(
                f"Opening {path}"
            )

            datasets.append(
                rasterio.open(path)
            )

        mosaic, transform = merge(
            datasets
        )

        profile = datasets[0].profile.copy()

        profile.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "compress": "deflate",
            }
        )

        output = os.path.join(
            RAW_DIR,
            "lroc_south_pole_merged.tif",
        )

        with rasterio.open(
            output,
            "w",
            **profile,
        ) as dst:

            dst.write(
                mosaic
            )

        print(
            f"\nMerged mosaic:\n{output}"
        )

        return output

    finally:

        for ds in datasets:

            ds.close()


# ============================================================
# CLIP RASTER
# ============================================================

def clip_raster(
    input_path,
    output_path,
):

    print()
    print("=" * 70)
    print("CLIPPING RASTER")
    print("=" * 70)

    print(
        f"Input:\n{input_path}"
    )

    print(
        f"Output:\n{output_path}"
    )

    with rasterio.open(
        input_path
    ) as src:

        print(
            f"\nCRS: {src.crs}"
        )

        print(
            f"Resolution: "
            f"{src.res}"
        )

        # ----------------------------------------------------
        # Convert geographic bbox to source CRS
        # ----------------------------------------------------

        bounds = transform_bounds(
            "EPSG:4326",
            src.crs,
            MIN_LON,
            MIN_LAT,
            MAX_LON,
            MAX_LAT,
            densify_pts=100,
        )

        print(
            "\nProjected bbox:"
        )

        print(
            f"  {bounds}"
        )

        window = from_bounds(
            *bounds,
            transform=src.transform,
        )

        window = (
            window
            .round_offsets()
            .round_lengths()
        )

        print(
            "\nPixel window:"
        )

        print(
            f"  {window}"
        )

        # ----------------------------------------------------
        # Read ONLY requested window
        # ----------------------------------------------------

        data = src.read(
            window=window
        )

        transform = (
            src.window_transform(
                window
            )
        )

        profile = src.profile.copy()

        profile.update(
            {
                "height": data.shape[1],
                "width": data.shape[2],
                "transform": transform,
                "compress": "deflate",
            }
        )

        with rasterio.open(
            output_path,
            "w",
            **profile,
        ) as dst:

            dst.write(
                data
            )

    print(
        "\nClip complete."
    )


# ============================================================
# LOLA REMOTE EXTRACTION
# ============================================================

def download_lola_bbox():

    """
    Try to open the USGS LOLA GeoTIFF directly over HTTP
    through GDAL/rasterio.

    If the server supports HTTP range requests, rasterio
    should retrieve only the blocks needed for the bbox.

    If not, we fail with an informative message rather than
    silently downloading 8 GB.
    """

    print()
    print("=" * 70)
    print("LOLA 118-M DEM")
    print("=" * 70)

    print(
        "\nOpening remote LOLA GeoTIFF:"
    )

    print(
        LOLA_URL
    )

    output = os.path.join(
        CLIPPED_DIR,
        "lola_118m_dem.tif",
    )

    # --------------------------------------------------------
    # GDAL virtual filesystem URL
    # --------------------------------------------------------

    vsi_url = (
        "/vsicurl/"
        + LOLA_URL
    )

    try:

        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
            GDAL_HTTP_USERAGENT=(
                "Mozilla/5.0"
            ),
        ):

            with rasterio.open(
                vsi_url
            ) as src:

                print(
                    f"\nLOLA CRS: {src.crs}"
                )

                print(
                    f"LOLA dimensions: "
                    f"{src.width} x {src.height}"
                )

                print(
                    f"LOLA resolution: "
                    f"{src.res}"
                )

                # ------------------------------------------------
                # LOLA is Simple Cylindrical / lon-lat
                # ------------------------------------------------

                bounds = transform_bounds(
                    "EPSG:4326",
                    src.crs,
                    MIN_LON,
                    MIN_LAT,
                    MAX_LON,
                    MAX_LAT,
                    densify_pts=100,
                )

                window = from_bounds(
                    *bounds,
                    transform=src.transform,
                )

                window = (
                    window
                    .round_offsets()
                    .round_lengths()
                )

                print(
                    "\nRequested LOLA window:"
                )

                print(
                    window
                )

                # ------------------------------------------------
                # Read only requested region
                # ------------------------------------------------

                data = src.read(
                    1,
                    window=window,
                )

                transform = (
                    src.window_transform(
                        window
                    )
                )

                profile = src.profile.copy()

                profile.update(
                    {
                        "driver": "GTiff",
                        "height": data.shape[0],
                        "width": data.shape[1],
                        "count": 1,
                        "transform": transform,
                        "compress": "deflate",
                        "dtype": data.dtype,
                    }
                )

                # ------------------------------------------------
                # IMPORTANT:
                #
                # LOLA is 16-bit signed integer with scale 0.5.
                #
                # Convert DN to actual elevation meters.
                # ------------------------------------------------

                data = (
                    data.astype(
                        np.float32
                    )
                    * 0.5
                )

                profile.update(
                    {
                        "dtype": "float32",
                        "nodata": -9999.0,
                    }
                )

                # Preserve invalid pixels
                if src.nodata is not None:

                    invalid = (
                        data
                        == (
                            float(src.nodata)
                            * 0.5
                        )
                    )

                    data[
                        invalid
                    ] = -9999.0

                with rasterio.open(
                    output,
                    "w",
                    **profile,
                ) as dst:

                    dst.write(
                        data,
                        1,
                    )

        print(
            "\nLOLA extraction complete:"
        )

        print(
            output
        )

        return output

    except Exception as e:

        print()
        print("=" * 70)
        print("REMOTE LOLA READ FAILED")
        print("=" * 70)

        print(
            f"\n{type(e).__name__}: {e}"
        )

        print(
            "\nYour network/GDAL combination could not "
            "window-read the remote LOLA GeoTIFF."
        )

        print(
            "\nThe full LOLA file is ~8 GB, so the script "
            "will NOT automatically download it."
        )

        print(
            "\nYou can manually download it from:"
        )

        print(
            LOLA_URL
        )

        return None


# ============================================================
# BBOX GEOJSON
# ============================================================

def save_bbox():

    print()
    print(
        "Saving bbox GeoJSON..."
    )

    # --------------------------------------------------------
    # Ordinary bbox
    # --------------------------------------------------------

    polygon = Polygon(
        [
            (MIN_LON, MIN_LAT),
            (MAX_LON, MIN_LAT),
            (MAX_LON, MAX_LAT),
            (MIN_LON, MAX_LAT),
            (MIN_LON, MIN_LAT),
        ]
    )

    gdf = gpd.GeoDataFrame(
        {
            "name": [
                "lunar_south_pole_bbox"
            ]
        },
        geometry=[
            polygon
        ],
        crs="EPSG:4326",
    )

    path = os.path.join(
        OUTPUT_DIR,
        "bbox.geojson",
    )

    gdf.to_file(
        path,
        driver="GeoJSON",
    )

    print(
        f"Saved:\n{path}"
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_raster_info(
    path,
    name,
):

    if path is None:
        return

    if not os.path.exists(path):
        return

    with rasterio.open(
        path
    ) as src:

        print()
        print(
            f"{name}"
        )

        print(
            "-" * 50
        )

        print(
            f"CRS       : {src.crs}"
        )

        print(
            f"Size      : "
            f"{src.width} x {src.height}"
        )

        print(
            f"Resolution: "
            f"{src.res}"
        )

        print(
            f"Bounds    : "
            f"{src.bounds}"
        )

        print(
            f"Dtype     : "
            f"{src.dtypes[0]}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("LUNAR SOUTH POLE DATASET BUILDER")
    print("=" * 70)

    print_bbox()

    # --------------------------------------------------------
    # Save bbox
    # --------------------------------------------------------

    save_bbox()

    # --------------------------------------------------------
    # LROC
    # --------------------------------------------------------

    try:

        lroc_tiles = (
            download_lroc()
        )

        lroc_merged = (
            merge_lroc_tiles(
                lroc_tiles
            )
        )

        lroc_output = os.path.join(
            CLIPPED_DIR,
            "lroc_nac_panchromatic.tif",
        )

        clip_raster(
            lroc_merged,
            lroc_output,
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("LROC FAILED")
        print("=" * 70)

        print(
            f"\n{type(e).__name__}: {e}"
        )

        print(
            "\nNo LOLA processing will be attempted "
            "until the LROC problem is fixed."
        )

        raise

    # --------------------------------------------------------
    # LOLA
    # --------------------------------------------------------

    lola_output = (
        download_lola_bbox()
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL DATASET")
    print("=" * 70)

    print_raster_info(
        lroc_output,
        "LROC NAC PANCHROMATIC",
    )

    print_raster_info(
        lola_output,
        "LOLA 118 m DEM",
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print(
        "Files:"
    )

    print(
        f"\nLROC:\n  {lroc_output}"
    )

    if lola_output:

        print(
            f"\nLOLA:\n  {lola_output}"
        )

    print(
        f"\nBBox:\n  "
        f"{os.path.join(OUTPUT_DIR, 'bbox.geojson')}"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()