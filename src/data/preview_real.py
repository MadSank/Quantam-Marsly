"""Render compact representative real-data patch diagnostics."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import rasterio
import yaml

from src.data.preprocess_real import horn_window, valid_dtm


def main() -> None:
    cfg=yaml.safe_load(Path("configs/real_data.yaml").read_text(encoding="utf-8"))
    processed=Path(cfg["paths"]["processed_dir"])
    df=pd.read_parquet(processed/"metadata.parquet")
    sources=pd.read_csv(cfg["paths"]["manifest_path"])
    sites=sources.set_index("site_id")
    out=processed/"diagnostics"; out.mkdir(parents=True,exist_ok=True)
    dataset_meta=json.loads((processed/"dataset_metadata.json").read_text(encoding="utf-8"))
    scales={item["site_id"]:item["native_resolution_m"] for item in dataset_meta["sites"]}
    for site_id,group in df.groupby("site_id"):
        source=sites.loc[site_id]
        ortho_path=Path(source.ortho_path); dtm_path=Path(source.dtm_path)
        band=int(source.ortho_band)
        with rasterio.open(ortho_path) as o, rasterio.open(dtm_path) as d:
            for label in (0,1):
                candidates=group[group.safe_label==label]
                if candidates.empty: continue
                rec=candidates.iloc[len(candidates)//2]
                row,col,L=int(rec.row),int(rec.col),int(rec.native_window_px)
                dtm=d.read(1,window=rasterio.windows.Window(col,row,L,L))
                image=o.read(band,window=rasterio.windows.Window(col,row,L,L))
                dvalid=valid_dtm(dtm,d.nodata); ivalid=image!=0; joint=dvalid&ivalid
                scale=float(scales[site_id])
                k15=max(1,int(round(15.0/scale))); halo=2*k15
                zh=d.read(1,window=rasterio.windows.Window(col-halo,row-halo,L+2*halo,L+2*halo),boundless=True,fill_value=d.nodata)
                slope=horn_window(zh,valid_dtm(zh,d.nodata),scale,5.0)
                slope=slope[2*k15:2*k15+L,2*k15:2*k15+L]
                shard=np.load(rec.shard_path,mmap_mode="r")[int(rec.shard_idx)].astype(np.float32)
                image_input=shard[0] if shard.ndim==3 else shard
                fig,axs=plt.subplots(2,3,figsize=(11,7),constrained_layout=True)
                axs[0,0].imshow(image_input,cmap="gray",vmin=0,vmax=1); axs[0,0].set_title("2 m image input (label calibrated)")
                axs[0,1].imshow(np.ma.masked_where(~dvalid,dtm),cmap="terrain"); axs[0,1].set_title("Native DTM (m)")
                sm=axs[0,2].imshow(np.ma.masked_invalid(slope),cmap="magma",vmin=0,vmax=30); axs[0,2].set_title("Native 5 m Horn slope (°)"); fig.colorbar(sm,ax=axs[0,2],shrink=.75)
                axs[1,0].imshow(shard[2] if shard.ndim==3 else ivalid,cmap="gray",vmin=0,vmax=1); axs[1,0].set_title("Ortho valid coverage (model channel)")
                axs[1,1].imshow(shard[1] if shard.ndim==3 else dvalid,cmap="gray",vmin=0,vmax=1); axs[1,1].set_title("DTM valid coverage (model channel)")
                axs[1,2].imshow(joint,cmap="gray",vmin=0,vmax=1); axs[1,2].set_title("Native joint valid mask")
                for ax in axs.flat: ax.set_xticks([]); ax.set_yticks([])
                fig.suptitle(f"{site_id} {rec.patch_id} | safe proxy={label} | coverage={rec.valid_coverage:.3f} | mean slope={rec.mean_slope_5m:.2f}°")
                fig.savefig(out/f"{site_id}_class{label}.png",dpi=125); plt.close(fig)
    print(out)


if __name__=="__main__": main()
