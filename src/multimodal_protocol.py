"""Frozen condition definitions for the formal LiDAR+mmWave study."""

from __future__ import annotations

from dataclasses import asdict, dataclass


MODALITY_MASKS = {
    "lidar_mmwave": (False, False, True, True),
    "lidar_only": (False, False, False, True),
    "mmwave_only": (False, False, True, False),
}
DEFAULT_SEVERITIES = (0.0, 0.25, 0.5, 0.75, 0.9)
DEFAULT_CORRUPTION_SEEDS = (7, 21, 42, 84, 168)
DEFAULT_GEOMETRIES = ("uniform", "azimuth_sector")


@dataclass(frozen=True)
class ExperimentCondition:
    """One immutable inference condition in the preregistered matrix."""

    mask_name: str
    geometry: str
    corruption_seed: int
    lidar_drop_rate: float
    mmwave_drop_rate: float

    @property
    def modality_mask(self) -> tuple[bool, bool, bool, bool]:
        try:
            return MODALITY_MASKS[self.mask_name]
        except KeyError as error:
            raise ValueError(f"Unknown modality mask: {self.mask_name}") from error

    @property
    def condition_id(self) -> str:
        return (
            f"{self.mask_name}__{self.geometry}__seed_{self.corruption_seed}"
            f"__lidar_{self.lidar_drop_rate:.2f}"
            f"__mmwave_{self.mmwave_drop_rate:.2f}"
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["condition_id"] = self.condition_id
        payload["modality_mask"] = list(self.modality_mask)
        return payload


def _validate_rates(severities: tuple[float, ...]) -> None:
    if not severities or severities[0] != 0.0:
        raise ValueError("severities must start with the clean rate 0.0")
    if tuple(sorted(set(severities))) != severities:
        raise ValueError("severities must be unique and increasing")
    if any(rate < 0.0 or rate >= 1.0 for rate in severities):
        raise ValueError("formal point-loss severities must be in [0, 1)")


def clean_baseline_conditions() -> list[ExperimentCondition]:
    return [
        ExperimentCondition(mask_name, "clean", 0, 0.0, 0.0)
        for mask_name in ("lidar_mmwave", "lidar_only", "mmwave_only")
    ]


def smoke_test_conditions() -> list[ExperimentCondition]:
    return clean_baseline_conditions() + [
        ExperimentCondition("lidar_mmwave", "uniform", 7, 0.5, 0.5),
        ExperimentCondition("lidar_mmwave", "azimuth_sector", 7, 0.5, 0.5),
    ]


def formal_conditions(
    severities: tuple[float, ...] = DEFAULT_SEVERITIES,
    corruption_seeds: tuple[int, ...] = DEFAULT_CORRUPTION_SEEDS,
    geometries: tuple[str, ...] = DEFAULT_GEOMETRIES,
) -> list[ExperimentCondition]:
    """Create the de-duplicated formal matrix.

    Unimodal conditions are evaluated once per sensor severity and are reused
    across the 5x5 fusion grid during quality-feature construction.
    """
    _validate_rates(severities)
    unsupported = set(geometries) - set(DEFAULT_GEOMETRIES)
    if unsupported:
        raise ValueError(f"Unsupported corruption geometries: {sorted(unsupported)}")
    if not corruption_seeds:
        raise ValueError("at least one corruption seed is required")

    conditions = clean_baseline_conditions()
    for geometry in geometries:
        for seed in corruption_seeds:
            for lidar_rate in severities:
                if lidar_rate > 0.0:
                    conditions.append(
                        ExperimentCondition(
                            "lidar_only", geometry, seed, lidar_rate, 0.0
                        )
                    )
            for mmwave_rate in severities:
                if mmwave_rate > 0.0:
                    conditions.append(
                        ExperimentCondition(
                            "mmwave_only", geometry, seed, 0.0, mmwave_rate
                        )
                    )
            for lidar_rate in severities:
                for mmwave_rate in severities:
                    if lidar_rate == 0.0 and mmwave_rate == 0.0:
                        continue
                    conditions.append(
                        ExperimentCondition(
                            "lidar_mmwave",
                            geometry,
                            seed,
                            lidar_rate,
                            mmwave_rate,
                        )
                    )

    ids = [condition.condition_id for condition in conditions]
    if len(ids) != len(set(ids)):
        raise RuntimeError("formal condition IDs are not unique")
    return conditions
