"""Create a separate real HiRISE dataset from exact companion PDS labels.

Reads source rasters by aligned pixel windows after validating the detached ortho
label against both product IDs and the DTM's embedded PDS projection. It never
uses the JP2 GeoJP2 transform/CRS when the exact label establishes the grid.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.windows import Window
import yaml

from src.data.preprocess import compute_tile_roughness
from src.data.splits import assign_grouped_folds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def pds_scalar(text: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^\r\n]+)", text, re.M | re.I)
    if not match:
        raise ValueError(f"Required PDS label field missing: {key}")
    return match.group(1).strip().strip('"')


def json_scalar(text: str, key: str) -> str:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"?([^",\r\n]+)', text, re.I)
    if not match:
        raise ValueError(f"Required embedded PDS field missing: {key}")
    return match.group(1).strip().strip('"')


def unit_value(text: str, key: str) -> float:
    m = re.search(rf"{re.escape(key)}\s*=\s*\{{\s*value\s*=\s*([+\-\d.eE]+)", text, re.I | re.S)
    if m:
        return float(m.group(1))
    return float(pds_scalar(text, key).split()[0])


def object_value(text: str, object_name: str, key: str) -> float:
    obj = re.search(rf"\"{re.escape(object_name)}\"\s*:\s*\{{(.*?)\n\s*\}}", text, re.S)
    if not obj:
        raise ValueError(f"Embedded PDS object missing: {object_name}")
    m = re.search(rf"\"{re.escape(key)}\"\s*:\s*([+\-\d.eE]+)", obj.group(1), re.I)
    if not m:
        raise ValueError(f"Embedded PDS field missing: {object_name}.{key}")
    return float(m.group(1))


def pds_label_grid(dtm: rasterio.DatasetReader) -> tuple[CRS, rasterio.Affine, dict[str, Any]]:
    tags = dtm.tags(ns="json:PDS")
    raw = next(iter(tags.values()), "")
    if not raw:
        raise ValueError("GDAL did not expose the embedded DTM PDS label")
    # Read projection object from GDAL's lossless JSON label.
    def num(key: str) -> float:
        m = re.search(rf'"{re.escape(key)}"\s*:\s*(?:\{{\s*"value"\s*:\s*)?([+\-\d.eE]+)', raw, re.I)
        if not m:
            raise ValueError(f"Embedded DTM PDS field missing: {key}")
        return float(m.group(1))

    scale = num("MAP_SCALE")
    lat = num("CENTER_LATITUDE")
    lon = num("CENTER_LONGITUDE")
    radius_m = num("A_AXIS_RADIUS") * 1000.0
    line_offset = num("LINE_PROJECTION_OFFSET")
    sample_offset = num("SAMPLE_PROJECTION_OFFSET")
    crs = CRS.from_proj4(
        f"+proj=eqc +lat_ts={lat} +lat_0=0 +lon_0={lon} +a={radius_m} +b={radius_m} +units=m +no_defs"
    )
    transform = rasterio.Affine(
        scale, 0.0, -(sample_offset + 0.5) * scale,
        0.0, -scale, (line_offset + 0.5) * scale,
    )
    meta = {
        "map_scale_m_per_pixel": scale, "center_latitude": lat,
        "center_longitude": lon, "radius_m": radius_m,
        "line_projection_offset": line_offset,
        "sample_projection_offset": sample_offset,
        "dtm_scale": num("SCALING_FACTOR"),
        "dtm_offset": num("OFFSET"),
        "elevation_unit": "m above Mars 2000 areoid",
        "nodata": dtm.nodata,
    }
    if abs(transform.a - dtm.transform.a) > 1e-7 or abs(transform.c - dtm.transform.c) > 1e-5 or abs(transform.f - dtm.transform.f) > 1e-5:
        raise ValueError("Label-derived DTM transform does not match GDAL DTM transform")
    if crs != dtm.crs:
        raise ValueError("Label-derived DTM CRS does not match embedded DTM CRS")
    return crs, transform, meta


def valid_dtm(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    mask = np.isfinite(arr) & (arr > -1e30)
    if nodata is not None:
        mask &= arr != nodata
    return mask


def horn_window(z: np.ndarray, valid: np.ndarray, scale: float, baseline_m: float) -> np.ndarray:
    k = max(1, int(round(baseline_m / scale)))
    h, w = z.shape
    out = np.full((h, w), np.nan, dtype=np.float32)
    if h <= 2 * k or w <= 2 * k:
        return out
    a = np.where(valid, z, np.nan).astype(np.float64, copy=False)
    nw, n, ne = a[:-2*k, :-2*k], a[:-2*k, k:-k], a[:-2*k, 2*k:]
    west, east = a[k:-k, :-2*k], a[k:-k, 2*k:]
    sw, s, se = a[2*k:, :-2*k], a[2*k:, k:-k], a[2*k:, 2*k:]
    dx = ((ne + 2*east + se) - (nw + 2*west + sw)) / (8*k*scale)
    dy = ((nw + 2*n + ne) - (sw + 2*s + se)) / (8*k*scale)
    result = np.degrees(np.arctan(np.hypot(dx, dy)))
    center_valid = valid[k:-k, k:-k]
    out[k:-k, k:-k] = np.where(center_valid, result, np.nan).astype(np.float32)
    return out


def read_window(dataset: rasterio.DatasetReader, row: int, col: int, height: int, width: int, band: int = 1) -> np.ndarray:
    return dataset.read(band, window=Window(col, row, width, height), boundless=True, fill_value=dataset.nodata or 0)


def extract_array(array: np.ndarray, row: int, col: int, height: int, width: int, fill: float) -> np.ndarray:
    """Small window view/copy from one in-memory source scene, padded only at edges."""
    out = np.full((height, width), fill, dtype=array.dtype)
    r0, c0 = max(row, 0), max(col, 0)
    r1, c1 = min(row + height, array.shape[0]), min(col + width, array.shape[1])
    if r1 > r0 and c1 > c0:
        out[r0-row:r1-row, c0-col:c1-col] = array[r0:r1, c0:c1]
    return out


def preprocess_site(row: pd.Series, root: Path, processed_dir: Path, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ortho_path = (root / row.ortho_path).resolve()
    dtm_path = (root / row.dtm_path).resolve()
    label_raw = os.path.expandvars(str(row.label_path).replace("%TEMP%", os.environ.get("TEMP", "")))
    label_path = Path(label_raw)
    band = int(row.ortho_band)
    label_text = label_path.read_text(encoding="utf-8", errors="replace")
    site_id = str(row.site_id)
    records: list[dict[str, Any]] = []
    kept: dict[int, list[tuple[dict[str, Any], np.ndarray]]] = {0: [], 1: []}
    seen_by_class = {0: 0, 1: 0}
    exclusions = {"grid_candidates":0,"excluded_joint_coverage":0,"excluded_dtm_coverage":0,"excluded_slope_support":0}
    max_per_class_cfg = cfg["preprocessing"].get("max_tiles_per_class_per_site", 750)
    max_per_class = None if max_per_class_cfg is None else int(max_per_class_cfg)
    sitewise_stretch = bool(cfg["preprocessing"].get("sitewise_image_stretch", True))
    rng = np.random.default_rng(7100 + int(site_id[-2:]))

    with rasterio.open(dtm_path) as d, rasterio.open(ortho_path) as o:
        d_tags = d.tags(ns="json:PDS")
        d_raw = next(iter(d_tags.values()), "")
        product = json_scalar(d_raw, "PRODUCT_ID")
        label_product = pds_scalar(label_text, "PRODUCT_ID")
        compressed = re.search(r"FILE_NAME\s*=\s*\"([^\"]+\.JP2)\"", label_text, re.I)
        src_ids = re.search(r"SOURCE_PRODUCT_ID\s*=\s*\((.*?)\)", label_text, re.I | re.S)
        expected_ortho = compressed.group(1).casefold() if compressed else ""
        expected_dtm = product.casefold()
        source_matches = bool(src_ids and expected_dtm in src_ids.group(1).casefold())
        if not expected_ortho or expected_ortho != ortho_path.name.casefold() or not source_matches:
            raise ValueError(f"{site_id}: detached label product/source IDs do not match exact files")
        if label_product.casefold() not in ortho_path.stem.casefold():
            raise ValueError(f"{site_id}: label PRODUCT_ID does not match the ortho filename")
        if d.width != o.width or d.height != o.height or int(pds_scalar(label_text, "LINE_SAMPLES")) != d.width or int(pds_scalar(label_text, "LINES")) != d.height:
            raise ValueError(f"{site_id}: dimensions differ from exact PDS label")
        if int(pds_scalar(label_text, "BANDS")) != o.count or band > o.count:
            raise ValueError(f"{site_id}: label band count or selected band is inconsistent")
        crs, transform, gmeta = pds_label_grid(d)
        scale = gmeta["map_scale_m_per_pixel"]
        native_span = 64 if scale > 1.5 else 128
        stride = native_span
        k5 = max(1, int(round(5.0 / scale)))
        k15 = max(1, int(round(15.0 / scale)))
        max_halo = 2 * k15
        core_null = int(pds_scalar(label_text, "CORE_NULL"))
        radiometric_scale = unit_value(label_text, "SCALING_FACTOR")
        radiometric_offset = unit_value(label_text, "OFFSET")
        low_sat = int(pds_scalar(label_text, "CORE_LOW_INSTR_SATURATION"))
        high_sat = int(pds_scalar(label_text, "CORE_HIGH_INSTR_SATURATION"))
        filters = re.search(r"FILTER_NAME\s*=\s*\((.*?)\)", label_text, re.I | re.S)
        filter_names = [x.strip().strip('"') for x in re.findall(r'"([^\"]+)"', filters.group(1))] if filters else [pds_scalar(label_text, "FILTER_NAME")]
        selected_filter = filter_names[band - 1] if len(filter_names) >= band else filter_names[0]
        if site_id == "site_04" and selected_filter.casefold() not in {"red", "visible red"}:
            raise ValueError(f"Coprates expected verified RED band 2; got {selected_filter}")

        if sitewise_stretch:
            # Legacy mode: per-site image normalization. Controlled evaluation
            # disables this and fits one scale using training-fold patches only.
            hist = np.zeros(256, dtype=np.int64)
            for r0 in range(0, d.height, 1024):
                hh = min(1024, d.height-r0)
                dn = read_window(o, r0, 0, hh, o.width, band=band)
                vals, cnt = np.unique(dn[dn != core_null], return_counts=True)
                keep = (vals >= 0) & (vals <= 255)
                hist[vals[keep].astype(int)] += cnt[keep]
            cdf = np.cumsum(hist)
            nvalid = int(cdf[-1])
            if nvalid == 0:
                raise ValueError(f"{site_id}: no valid image pixels")
            lo_dn = int(np.searchsorted(cdf, 0.02*nvalid))
            hi_dn = int(np.searchsorted(cdf, 0.98*nvalid))
            lo_if = lo_dn * radiometric_scale + radiometric_offset
            hi_if = hi_dn * radiometric_scale + radiometric_offset
            if hi_if <= lo_if:
                raise ValueError(f"{site_id}: invalid radiometric stretch")
        else:
            lo_dn = hi_dn = None
            lo_if, hi_if = 0.0, 1.0

        # Load only this one source scene, with no scene-sized resampled copies
        # or derivative stacks. Largest pair is ~1.1 GiB (float32 DTM + uint8 JP2).
        dtm_array = d.read(1)
        ortho_array = o.read(band)

        for r in range(0, d.height-native_span+1, stride):
            for c in range(0, d.width-native_span+1, stride):
                exclusions["grid_candidates"] += 1
                patch_z = dtm_array[r:r+native_span, c:c+native_span]
                patch_ortho = ortho_array[r:r+native_span, c:c+native_span]
                dvalid = valid_dtm(patch_z, d.nodata)
                ivalid = patch_ortho != core_null
                joint = dvalid & ivalid
                coverage = float(joint.mean())
                if dvalid.mean() < 0.95:
                    exclusions["excluded_dtm_coverage"] += 1
                    continue
                if coverage < 0.95:
                    exclusions["excluded_joint_coverage"] += 1
                    continue
                # Native DTM derivative patches with sufficient halo so no
                # artificial patch-edge slope enters the target.
                rr0, cc0 = r-max_halo, c-max_halo
                hread = native_span + 2*max_halo
                zhalo = extract_array(dtm_array, rr0, cc0, hread, hread, d.nodata or -3.4028235e38)
                vhalo = valid_dtm(zhalo, d.nodata)
                sl5 = horn_window(zhalo, vhalo, scale, 5.0)
                sl15 = horn_window(zhalo, vhalo, scale, 15.0)
                center = (slice(2*k15, 2*k15+native_span), slice(2*k15, 2*k15+native_span))
                slope5 = sl5[center]
                slope15 = sl15[center]
                f5 = dvalid & np.isfinite(slope5)
                f15 = dvalid & np.isfinite(slope15)
                if f5.sum() < 0.95*dvalid.sum():
                    exclusions["excluded_slope_support"] += 1
                    continue
                mean5 = float(np.mean(slope5[f5]))
                mean15 = float(np.mean(slope15[f15])) if f15.any() else mean5
                roughness = compute_tile_roughness(patch_z, dvalid, scale)
                hazard = float(np.mean(slope5[f5] > 15.0))
                safe = int((mean5 < 8.0) and (hazard < 0.05))
                seen_by_class[safe] += 1

                calibrated = patch_ortho.astype(np.float32)*radiometric_scale+radiometric_offset
                if native_span == 128:
                    calibrated = calibrated.reshape(64,2,64,2)
                    valid4 = ivalid.reshape(64,2,64,2)
                    dvalid4 = dvalid.reshape(64,2,64,2)
                    denom = valid4.sum(axis=(1,3))
                    image_2m = np.divide((calibrated*valid4).sum(axis=(1,3)), denom, out=np.zeros((64,64),np.float32), where=denom>0)
                    cell_valid = denom > 0
                    cell_coverage = denom.astype(np.float32)/4.0
                    dtm_coverage = dvalid4.mean(axis=(1,3)).astype(np.float32)
                else:
                    image_2m = calibrated
                    cell_valid = ivalid
                    cell_coverage = ivalid.astype(np.float32)
                    dtm_coverage = dvalid.astype(np.float32)
                if sitewise_stretch:
                    image_2m = np.clip((image_2m-lo_if)/(hi_if-lo_if),0,1).astype(np.float32)
                else:
                    image_2m = image_2m.astype(np.float32)
                image_2m[~cell_valid] = 0.0
                model_patch = np.stack([image_2m, dtm_coverage, cell_coverage]).astype(np.float32)
                # In controlled mode max_per_class=None retains every valid
                # candidate. Any cap in legacy mode is training-data sampling;
                # controlled folds must never cap validation/test by label.
                manifest_row = {
                    "site_id":site_id,"patch_id":f"{site_id}_r{r}_c{c}",
                    "tile_id":f"{site_id}_r{r}_c{c}","image_id":site_id,
                    "row":r,"col":c,"native_window_px":native_span,
                    "patch_gsd_m":2.0,"footprint_m":128.0,
                    "valid_coverage":coverage,"dtm_valid_coverage":float(dvalid.mean()),
                    "image_valid_coverage":float(ivalid.mean()),
                    "resampled_cell_valid_fraction":float(cell_coverage.mean()),
                    "mean_slope_5m":mean5,"mean_slope_15m":mean15,
                    "roughness_m":roughness,"hazard_fraction":hazard,"safe_label":safe,
                    "filter":selected_filter,"ortho_band":band,
                    "radiometric_scale":radiometric_scale,"radiometric_offset":radiometric_offset,
                    "p2_dn":lo_dn,"p98_dn":hi_dn,"crs_wkt":crs.to_wkt(),
                    "transform":list(transform)[:6],"dtm_product_id":product,
                    "ortho_product_id":label_product,
                }
                queue=kept[safe]
                item=(manifest_row,model_patch)
                if max_per_class is None or len(queue)<max_per_class:
                    queue.append(item)
                else:
                    # Reservoir replacement maintains a seeded uniform sample.
                    j = int(rng.integers(0, seen_by_class[safe]))
                    if j < max_per_class:
                        queue[j]=item
        chosen=kept[0]+kept[1]
        if not chosen:
            raise ValueError(f"{site_id}: no usable patches at >=95% valid coverage")
        # Stable site-specific ordering; downstream shard index and metadata agree.
        chosen.sort(key=lambda x:(x[0]["row"],x[0]["col"]))
        arrays=np.stack([x[1] for x in chosen]).astype(np.float16)
        shard_dir=processed_dir/"shards"; shard_dir.mkdir(parents=True,exist_ok=True)
        shard_path=shard_dir/f"{site_id}_tiles.npy"
        np.save(shard_path,arrays)
        for i,(rec,_) in enumerate(chosen):
            rec["shard_path"]=str(shard_path); rec["shard_idx"]=i
            records.append(rec)
        stats={
            "site_id":site_id,"dtm_product_id":product,"ortho_product_id":label_product,
            "dimensions":[d.height,d.width],"native_resolution_m":scale,
            "selected_band":band,"selected_filter":selected_filter,
            "radiometric_equation":"I/F = DN * scaling_factor + offset",
            "scaling_factor":radiometric_scale,"offset":radiometric_offset,
            "display_stretch_dn":[lo_dn,hi_dn],"core_null":core_null,
            "low_saturation_dn":low_sat,"high_saturation_dn":high_sat,
            "candidate_count_by_safe_class_capped":{str(k):len(v) for k,v in kept.items()},
            "accepted_before_class_cap_by_safe_label":{str(k):v for k,v in seen_by_class.items()},
            **exclusions,
            "class_cap_excluded":sum(seen_by_class.values())-len(chosen),
            "sitewise_image_stretch":sitewise_stretch,
            "valid_fraction_in_retained_patches":float(np.mean([x[0]["valid_coverage"] for x in chosen])),
            "label_crs_and_transform_used":True,
        }
    return records,stats


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/real_data.yaml")
    args=parser.parse_args()
    root=Path.cwd()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest=pd.read_csv(root/cfg["paths"]["manifest_path"])
    out=Path(cfg["paths"]["processed_dir"]); out.mkdir(parents=True,exist_ok=True)
    all_rows=[]; stats=[]
    for _,row in manifest.iterrows():
        logging.info("Processing %s",row.site_id)
        records,site_stats=preprocess_site(row,root,out,cfg)
        all_rows.extend(records); stats.append(site_stats)
        logging.info("%s retained %d patches",row.site_id,len(records))
    df=pd.DataFrame(all_rows)
    if cfg.get("controlled_evaluation", False):
        site_order = cfg["splits"].get("site_order", sorted(df.site_id.astype(str).unique()))
        if set(site_order) != set(df.site_id.astype(str).unique()):
            raise ValueError("splits.site_order must list every processed site exactly once")
        site_to_fold = {site: idx for idx, site in enumerate(site_order)}
        df["fold"] = df.site_id.astype(str).map(site_to_fold).astype(int)
    else:
        df=assign_grouped_folds(df,n_splits=5,seed=int(cfg["splits"]["split_seed"]))
    df.to_parquet(out/"metadata.parquet",index=False)
    df.drop(columns=["crs_wkt"]).to_csv(out/"patch_manifest.csv",index=False)
    summary={"dataset":"real_hirise_v1","config":str(args.config),"sites":stats,
             "patch_count":len(df),"class_counts":{str(k):int(v) for k,v in df.safe_label.value_counts().sort_index().items()},
             "site_class_counts":{site:{str(k):int(v) for k,v in group.safe_label.value_counts().sort_index().items()} for site,group in df.groupby("site_id")},
             "split_image_ids":{str(f):sorted(df.loc[df.fold==f,"site_id"].unique().tolist()) for f in range(5)},
             "method":"native DTM derivatives, 128 m windows, mask-aware imagery averaging to 64x64 at nominal 2 m/px; model channels are calibrated image, DTM-valid coverage, image-valid coverage; no 0.5 m upsampling",
             "controlled_evaluation":bool(cfg.get("controlled_evaluation",False)),
             "site_fold_order":cfg.get("splits",{}).get("site_order")}
    (out/"dataset_metadata.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    logging.info("Wrote %s",out)


if __name__=="__main__":
    main()
