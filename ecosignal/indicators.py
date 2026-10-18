"""Detection-derived biodiversity indicators with explicit input validation."""
import numpy as np
import pandas as pd


def build_indicators(metadata, detections, traits):
    required_det = {"site", "year", "taxon", "detected"}
    required_meta = {"site", "year"}
    required_traits = {"taxon", "thermal_affinity"}
    if not required_det.issubset(detections.columns):
        raise ValueError("detections: missing required columns")
    if not required_meta.issubset(metadata.columns):
        raise ValueError("metadata: missing required columns")
    if not required_traits.issubset(traits.columns):
        raise ValueError("traits: missing required columns")
    if metadata.duplicated(["site", "year"]).any():
        raise ValueError("Duplicate site-year metadata")
    if detections.duplicated(["site", "year", "taxon"]).any():
        raise ValueError("Duplicate site-year-taxon")
    if traits.duplicated("taxon").any():
        raise ValueError("Duplicate traits")
    if detections[list(required_det)].isna().any().any():
        raise ValueError("Missing detection identifiers or values")
    if not detections["detected"].isin([0, 1]).all():
        raise ValueError("detected must be 0/1")
    if metadata[list(required_meta)].isna().any().any():
        raise ValueError("Missing site-year metadata identifiers")
    if traits[["taxon", "thermal_affinity"]].isna().any().any():
        raise ValueError("Missing thermal affinity or trait identifiers")
    if not np.isfinite(pd.to_numeric(traits.thermal_affinity, errors="coerce")).all():
        raise ValueError("Thermal affinities must be finite numeric values")

    observed_taxa = set(detections.taxon)
    trait_taxa = set(traits.taxon)
    missing_traits = observed_taxa - trait_taxa
    if missing_traits:
        raise ValueError(f"Missing thermal affinities for {len(missing_traits)} taxa: {sorted(missing_traits)[:5]}")
    keys = metadata[["site", "year"]].merge(
        detections[["site", "year"]].drop_duplicates(),
        on=["site", "year"], how="outer", indicator=True,
    )
    if (keys._merge != "both").any():
        raise ValueError("Metadata and detections must have identical site-year coverage")

    wide = detections.pivot(index=["site", "year"], columns="taxon", values="detected")
    if wide.isna().any().any():
        raise ValueError("Incomplete detection matrix; do not silently turn missing taxa into absence")

    x = detections.merge(traits[["taxon", "thermal_affinity"]], on="taxon", validate="many_to_one")
    x["weighted_affinity"] = x.detected * x.thermal_affinity
    totals = x.groupby(["site", "year"], as_index=False).agg(
        richness=("detected", "sum"), affinity_sum=("weighted_affinity", "sum")
    )
    totals["cti"] = totals.affinity_sum / totals.richness.replace(0, np.nan)
    totals = totals.drop(columns="affinity_sum")

    turnover = []
    for site, group in wide.groupby(level="site"):
        previous = None
        for (_, year), row in group.sort_index(level="year").iterrows():
            current = row.to_numpy(dtype=bool)
            if previous is None:
                dissimilarity = np.nan
            else:
                union = (current | previous).sum()
                dissimilarity = (current ^ previous).sum() / union if union else np.nan
            turnover.append({"site": site, "year": year, "jaccard_turnover": dissimilarity})
            previous = current
    return metadata.merge(totals, on=["site", "year"], validate="one_to_one").merge(
        pd.DataFrame(turnover), on=["site", "year"], validate="one_to_one"
    )


def reference_alerts(df, metric="cti", baseline_end=2009, quantile=0.95):
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    baseline = df.loc[df.year <= baseline_end, metric].dropna()
    if baseline.empty:
        raise ValueError("No observations in reference period")
    cutoff = float(baseline.quantile(quantile))
    out = df.copy()
    out[f"{metric}_reference_threshold"] = cutoff
    # Missing CTI is not an observed non-alert.
    out[f"{metric}_alert"] = pd.array(
        np.where(out[metric].isna(), pd.NA, out[metric] > cutoff), dtype="boolean"
    ).astype("Int64")
    return out
