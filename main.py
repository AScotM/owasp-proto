import argparse
import csv
import json
import logging
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


LOGGER = logging.getLogger("owasp-proto")
CATEGORIES = (
    "BROKEN_ACCESS_CONTROL",
    "CRYPTO_FAILURES",
    "INJECTION",
    "INSECURE_DESIGN",
    "SECURITY_MISCONFIGURATION",
    "VULNERABLE_COMPONENTS",
    "AUTH_FAILURES",
    "INTEGRITY_FAILURES",
    "LOGGING_MONITORING_FAILURES",
    "SSRF",
)


@dataclass(frozen=True)
class Finding:
    x: float
    y: float
    risk: float
    category: str


@dataclass(frozen=True)
class Prediction:
    x: float
    y: float
    risk: float
    neighbors: int = 0
    nearest_distance: float = 0.0
    local_spread: float = 0.0


@dataclass(frozen=True)
class Bounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def validate(self) -> None:
        values = (self.xmin, self.xmax, self.ymin, self.ymax)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Bounds must be finite")
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError("Bounds must have positive width and height")


@dataclass(frozen=True)
class ValidationResult:
    rmse: float
    mae: float
    bias: float
    folds: int
    observations: int
    power: float
    neighbors: Optional[int]
    seed: int


@dataclass(frozen=True)
class ModelOptions:
    power: float = 2.0
    neighbors: Optional[int] = 12
    epsilon: float = 1e-12
    scale_x: float = 1.0
    scale_y: float = 1.0
    chunk_size: int = 4096

    def validate(self) -> None:
        if not math.isfinite(self.power) or self.power <= 0:
            raise ValueError("Power must be finite and positive")
        if self.neighbors is not None and self.neighbors < 1:
            raise ValueError("Neighbors must be positive")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Epsilon must be finite and positive")
        if not all(math.isfinite(v) and v > 0 for v in (self.scale_x, self.scale_y)):
            raise ValueError("Coordinate scales must be finite and positive")
        if self.chunk_size < 1:
            raise ValueError("Chunk size must be positive")


class Dataset:
    def __init__(self, findings: Optional[Sequence[Finding]] = None):
        self.findings = list(findings or [])
        self.validate()

    def validate(self) -> None:
        for index, finding in enumerate(self.findings):
            if not all(math.isfinite(value) for value in (finding.x, finding.y, finding.risk)):
                raise ValueError(f"Nonfinite value at finding {index}")
            if not 0 <= finding.risk <= 10:
                raise ValueError(f"Risk outside 0..10 at finding {index}")
            if not finding.category.strip():
                raise ValueError(f"Empty category at finding {index}")

    @classmethod
    def from_csv(cls, filename: str, strict: bool = True) -> "Dataset":
        findings = []
        with open(filename, newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"x", "y", "risk", "category"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"Input CSV requires columns: {', '.join(sorted(required))}")
            for row_number, row in enumerate(reader, start=2):
                try:
                    finding = Finding(
                        x=float(row["x"]),
                        y=float(row["y"]),
                        risk=float(row["risk"]),
                        category=str(row["category"]).strip(),
                    )
                    cls([finding])
                    findings.append(finding)
                except (TypeError, ValueError) as exc:
                    if strict:
                        raise ValueError(f"Invalid CSV row {row_number}: {exc}") from exc
                    LOGGER.warning("Skipping row %d: %s", row_number, exc)
        return cls(findings)

    def to_csv(self, filename: str) -> None:
        target = Path(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("x", "y", "risk", "category"))
            for finding in self.findings:
                writer.writerow((finding.x, finding.y, finding.risk, finding.category))

    @classmethod
    def synthetic(cls, count: int, seed: int, noise: float = 0.35) -> "Dataset":
        if count < 1 or not math.isfinite(noise) or noise < 0:
            raise ValueError("Invalid sample count or noise")
        rng = random.Random(seed)
        findings = []
        for _ in range(count):
            x = rng.uniform(0, 100)
            y = rng.uniform(0, 100)
            signal = 3.2 + 0.028 * x + 0.016 * y + 0.8 * math.sin(x / 15) + 0.6 * math.cos(y / 17)
            risk = min(10.0, max(0.0, signal + rng.gauss(0, noise)))
            findings.append(Finding(x, y, risk, rng.choice(CATEGORIES)))
        return cls(findings)

    def arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        points = np.array([(f.x, f.y) for f in self.findings], dtype=float).reshape(-1, 2)
        risks = np.array([f.risk for f in self.findings], dtype=float)
        return points, risks

    def bounds(self) -> Bounds:
        if not self.findings:
            raise ValueError("No findings")
        points, _ = self.arrays()
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5
        if ymin == ymax:
            ymin -= 0.5
            ymax += 0.5
        return Bounds(float(xmin), float(xmax), float(ymin), float(ymax))

    def statistics(self) -> dict:
        if not self.findings:
            return {"count": 0, "categories": {}}
        risks = [finding.risk for finding in self.findings]
        categories = defaultdict(list)
        for finding in self.findings:
            categories[finding.category].append(finding.risk)
        return {
            "count": len(risks),
            "mean": statistics.fmean(risks),
            "std_dev": statistics.pstdev(risks),
            "minimum": min(risks),
            "maximum": max(risks),
            "median": statistics.median(risks),
            "bounds": asdict(self.bounds()),
            "categories": {
                category: {
                    "count": len(values),
                    "mean": statistics.fmean(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
                for category, values in sorted(categories.items())
            },
        }

    def duplicates(self) -> dict:
        groups = defaultdict(list)
        for finding in self.findings:
            groups[(finding.x, finding.y)].append(finding.risk)
        duplicated = {key: values for key, values in groups.items() if len(values) > 1}
        return {
            "coordinate_groups": len(duplicated),
            "additional_observations": sum(len(values) - 1 for values in duplicated.values()),
            "conflicting_groups": sum(1 for values in duplicated.values() if max(values) != min(values)),
        }


class IDWModel:
    def __init__(self, options: ModelOptions):
        options.validate()
        self.options = options
        self.tree: Optional[cKDTree] = None
        self.points = np.empty((0, 2), dtype=float)
        self.values = np.empty(0, dtype=float)

    def fit(self, dataset: Dataset) -> "IDWModel":
        if not dataset.findings:
            raise ValueError("Cannot fit an empty dataset")
        points, values = dataset.arrays()
        self.points = points * np.array([self.options.scale_x, self.options.scale_y])
        self.values = values
        self.tree = cKDTree(self.points)
        return self

    def _predict_row(self, distances: np.ndarray, indices: np.ndarray) -> Tuple[float, int, float, float]:
        valid = np.isfinite(distances) & (indices >= 0) & (indices < len(self.values))
        distances = distances[valid]
        indices = indices[valid]
        if len(distances) == 0:
            raise ValueError("No neighbors returned")
        exact = distances <= self.options.epsilon
        if exact.any():
            risks = self.values[indices[exact]]
            return float(np.mean(risks)), len(risks), 0.0, float(np.std(risks))
        scaled = distances / np.min(distances)
        weights = np.power(scaled, -self.options.power)
        weights /= np.sum(weights)
        risks = self.values[indices]
        mean = float(np.dot(weights, risks))
        spread = float(np.sqrt(np.dot(weights, np.square(risks - mean))))
        return mean, len(risks), float(np.min(distances)), spread

    def predict(self, targets: np.ndarray) -> List[Prediction]:
        if self.tree is None:
            raise ValueError("Model is not fitted")
        targets = np.asarray(targets, dtype=float).reshape(-1, 2)
        if not np.isfinite(targets).all():
            raise ValueError("Targets must be finite")
        if len(targets) == 0:
            return []
        count = len(self.values) if self.options.neighbors is None else min(self.options.neighbors, len(self.values))
        result = []
        scale = np.array([self.options.scale_x, self.options.scale_y])
        for start in range(0, len(targets), self.options.chunk_size):
            chunk = targets[start:start + self.options.chunk_size]
            distances, indices = self.tree.query(chunk * scale, k=count, workers=1)
            distances = np.asarray(distances).reshape(len(chunk), count)
            indices = np.asarray(indices).reshape(len(chunk), count)
            for target, dist, idx in zip(chunk, distances, indices):
                risk, neighbors, nearest, spread = self._predict_row(dist, idx)
                result.append(Prediction(float(target[0]), float(target[1]), min(10.0, max(0.0, risk)), neighbors, nearest, spread))
        return result

    def predict_one(self, x: float, y: float) -> Prediction:
        return self.predict(np.array([[x, y]], dtype=float))[0]

    def predict_grid(self, bounds: Bounds, resolution: int) -> List[Prediction]:
        bounds.validate()
        if resolution < 1 or resolution > 2000:
            raise ValueError("Resolution must be in 1..2000")
        xs = np.linspace(bounds.xmin, bounds.xmax, resolution + 1)
        ys = np.linspace(bounds.ymin, bounds.ymax, resolution + 1)
        return self.predict(np.array([(x, y) for x in xs for y in ys], dtype=float))


class Evaluator:
    @staticmethod
    def folds(size: int, fold_count: int, seed: int) -> List[np.ndarray]:
        if size < 2:
            raise ValueError("At least two findings are required for cross-validation")
        if fold_count < 2:
            raise ValueError("At least two folds are required")
        indices = np.random.default_rng(seed).permutation(size)
        return [part for part in np.array_split(indices, min(size, fold_count)) if len(part)]

    @classmethod
    def cross_validate(cls, dataset: Dataset, options: ModelOptions, folds: int, seed: int) -> ValidationResult:
        partitions = cls.folds(len(dataset.findings), folds, seed)
        errors = []
        for test_indices in partitions:
            test_set = set(int(index) for index in test_indices)
            train = Dataset([finding for index, finding in enumerate(dataset.findings) if index not in test_set])
            model = IDWModel(options).fit(train)
            targets = np.array([(dataset.findings[int(i)].x, dataset.findings[int(i)].y) for i in test_indices])
            predictions = model.predict(targets)
            for index, prediction in zip(test_indices, predictions):
                errors.append(prediction.risk - dataset.findings[int(index)].risk)
        values = np.array(errors)
        return ValidationResult(
            rmse=float(np.sqrt(np.mean(values ** 2))),
            mae=float(np.mean(np.abs(values))),
            bias=float(np.mean(values)),
            folds=len(partitions),
            observations=len(errors),
            power=options.power,
            neighbors=options.neighbors,
            seed=seed,
        )

    @classmethod
    def optimize(cls, dataset: Dataset, powers: Sequence[float], neighbors: Sequence[Optional[int]], folds: int, seed: int, base: ModelOptions) -> dict:
        trials = []
        for power in powers:
            for count in neighbors:
                options = ModelOptions(power, count, base.epsilon, base.scale_x, base.scale_y, base.chunk_size)
                result = cls.cross_validate(dataset, options, folds, seed)
                trials.append(asdict(result))
        trials.sort(key=lambda entry: (entry["rmse"], entry["mae"], entry["power"]))
        return {"best": trials[0], "trials": trials}


class SpatialAnalysis:
    @staticmethod
    def morans_i(dataset: Dataset, neighbors: int = 5) -> Optional[dict]:
        if len(dataset.findings) < 3:
            return None
        points, values = dataset.arrays()
        centered = values - np.mean(values)
        denominator = float(np.dot(centered, centered))
        if denominator <= 0:
            return None
        tree = cKDTree(points)
        k = min(len(values), max(2, neighbors + 1))
        distances, indices = tree.query(points, k=k)
        numerator = 0.0
        weight_total = 0.0
        for i in range(len(values)):
            for distance, j in zip(np.atleast_1d(distances[i]), np.atleast_1d(indices[i])):
                if int(j) == i or not math.isfinite(float(distance)) or distance <= 0:
                    continue
                weight = 1.0 / float(distance)
                numerator += weight * float(centered[i] * centered[int(j)])
                weight_total += weight
        if weight_total == 0:
            return None
        score = len(values) * numerator / (weight_total * denominator)
        return {"morans_i": float(score), "expected_under_randomness": -1.0 / (len(values) - 1), "weight_sum": weight_total, "neighbors": k - 1}

    @staticmethod
    def block_average(dataset: Dataset, bounds: Bounds, block_size: float) -> List[dict]:
        bounds.validate()
        if not math.isfinite(block_size) or block_size <= 0:
            raise ValueError("Block size must be positive")
        buckets = defaultdict(list)
        nx = max(1, math.ceil((bounds.xmax - bounds.xmin) / block_size))
        ny = max(1, math.ceil((bounds.ymax - bounds.ymin) / block_size))
        for finding in dataset.findings:
            if not (bounds.xmin <= finding.x <= bounds.xmax and bounds.ymin <= finding.y <= bounds.ymax):
                continue
            bx = min(nx - 1, int((finding.x - bounds.xmin) / block_size))
            by = min(ny - 1, int((finding.y - bounds.ymin) / block_size))
            buckets[(bx, by)].append(finding.risk)
        return [
            {"block_x": bx, "block_y": by, "center_x": bounds.xmin + (bx + 0.5) * block_size,
             "center_y": bounds.ymin + (by + 0.5) * block_size, "count": len(values),
             "mean_risk": statistics.fmean(values), "minimum": min(values), "maximum": max(values)}
            for (bx, by), values in sorted(buckets.items())
        ]


class ReportWriter:
    @staticmethod
    def write_json(filename: str, payload: dict) -> None:
        target = Path(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False, sort_keys=True)
            stream.write("\n")

    @staticmethod
    def write_predictions(filename: str, predictions: Sequence[Prediction]) -> None:
        target = Path(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("x", "y", "predicted_risk", "neighbors", "nearest_distance", "local_spread"))
            for prediction in predictions:
                writer.writerow((prediction.x, prediction.y, prediction.risk, prediction.neighbors, prediction.nearest_distance, prediction.local_spread))

    @staticmethod
    def display(payload: dict) -> None:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


class OwaspRiskMap:
    def __init__(self):
        self.dataset = Dataset()
        self.findings = self.dataset.findings

    def configure_logging(self, level: int = logging.INFO) -> None:
        logging.basicConfig(level=level, format="%(levelname)s %(message)s")

    def load_csv(self, filename: str, reset: bool = False, strict: bool = True) -> None:
        loaded = Dataset.from_csv(filename, strict)
        findings = loaded.findings if reset else self.findings + loaded.findings
        self.dataset = Dataset(findings)
        self.findings = self.dataset.findings

    def generate_sample_data(self, n_points: int = 250, random_seed: Optional[int] = None) -> None:
        self.dataset = Dataset.synthetic(n_points, 42 if random_seed is None else random_seed)
        self.findings = self.dataset.findings

    def statistics_summary(self) -> dict:
        return self.dataset.statistics()

    def category_summary(self) -> List[dict]:
        stats = self.dataset.statistics()["categories"]
        return [{"category": name, **values} for name, values in stats.items()]

    def morans_i(self) -> Optional[float]:
        result = SpatialAnalysis.morans_i(self.dataset)
        return None if result is None else result["morans_i"]

    def idw_risk(self, target_x: float, target_y: float, power: float = 2, max_points: Optional[int] = None, use_optimized: bool = True) -> float:
        return IDWModel(ModelOptions(power, max_points)).fit(self.dataset).predict_one(target_x, target_y).risk

    def cross_validate(self, power: float = 2, k_folds: int = 5, use_optimized: bool = True, max_points: Optional[int] = None, random_seed: Optional[int] = None) -> ValidationResult:
        return Evaluator.cross_validate(self.dataset, ModelOptions(power, max_points), k_folds, 42 if random_seed is None else random_seed)

    def optimize_power(self, k_folds: int = 5, use_optimized: bool = True, max_points: Optional[int] = None, random_seed: int = 42) -> float:
        result = Evaluator.optimize(self.dataset, [1, 2, 3, 4, 5], [max_points], k_folds, random_seed, ModelOptions())
        return float(result["best"]["power"])

    def predict_grid(self, xmin: float, xmax: float, ymin: float, ymax: float, resolution: int, power: float = 2, max_points: Optional[int] = None, use_optimized: bool = True, show_progress: bool = False) -> List[Prediction]:
        return IDWModel(ModelOptions(power, max_points)).fit(self.dataset).predict_grid(Bounds(xmin, xmax, ymin, ymax), resolution)

    def predict_grid_lazy(self, xmin: float, xmax: float, ymin: float, ymax: float, resolution: int, power: float = 2, max_points: Optional[int] = None, use_optimized: bool = True) -> Iterator[Prediction]:
        model = IDWModel(ModelOptions(power, max_points)).fit(self.dataset)
        bounds = Bounds(xmin, xmax, ymin, ymax)
        bounds.validate()
        if resolution < 1 or resolution > 2000:
            raise ValueError("Resolution must be in 1..2000")
        for x in np.linspace(xmin, xmax, resolution + 1):
            for y in np.linspace(ymin, ymax, resolution + 1):
                yield model.predict_one(float(x), float(y))

    def save_csv(self, filename: str, predictions: Sequence[Prediction]) -> None:
        ReportWriter.write_predictions(filename, predictions)

    def block_average(self, xmin: float, xmax: float, ymin: float, ymax: float, block_size: float = 10) -> List[dict]:
        return SpatialAnalysis.block_average(self.dataset, Bounds(xmin, xmax, ymin, ymax), block_size)


def parse_neighbors(value: str) -> Optional[int]:
    if value.lower() in ("all", "none"):
        return None
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Neighbors must be positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="owasp-proto", description="Spatial risk interpolation and exploratory analysis")
    parser.add_argument("--input", help="CSV containing x,y,risk,category")
    parser.add_argument("--sample", type=int, default=0, help="Generate synthetic findings")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--neighbors", type=parse_neighbors, default=12)
    parser.add_argument("--scale-x", type=float, default=1.0)
    parser.add_argument("--scale-y", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--resolution", type=int, default=20)
    parser.add_argument("--bounds", nargs=4, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    parser.add_argument("--predict", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--cross-validate", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--powers", nargs="+", type=float, default=[1, 2, 3, 4, 5])
    parser.add_argument("--grid-output")
    parser.add_argument("--report-output")
    parser.add_argument("--sample-output")
    parser.add_argument("--block-size", type=float, default=20.0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict:
    if args.input and args.sample:
        raise ValueError("Use either --input or --sample")
    if not args.input and not args.sample:
        raise ValueError("Provide --input or --sample")
    dataset = Dataset.from_csv(args.input, strict=not args.skip_invalid) if args.input else Dataset.synthetic(args.sample, args.seed)
    if not dataset.findings:
        raise ValueError("No usable findings")
    options = ModelOptions(args.power, args.neighbors, scale_x=args.scale_x, scale_y=args.scale_y, chunk_size=args.chunk_size)
    options.validate()
    if args.sample_output:
        dataset.to_csv(args.sample_output)
    bounds = Bounds(*args.bounds) if args.bounds else dataset.bounds()
    bounds.validate()
    report = {
        "tool": "owasp-proto",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "imported" if args.input else "synthetic",
        "interpretation": "Interpolated scores are estimates, not confirmed vulnerabilities or calibrated probabilities",
        "statistics": dataset.statistics(),
        "duplicates": dataset.duplicates(),
        "spatial_autocorrelation": SpatialAnalysis.morans_i(dataset),
        "block_averages": SpatialAnalysis.block_average(dataset, bounds, args.block_size),
        "model": asdict(options),
    }
    if args.cross_validate:
        report["cross_validation"] = asdict(Evaluator.cross_validate(dataset, options, args.folds, args.seed))
    if args.optimize:
        report["optimization"] = Evaluator.optimize(dataset, args.powers, [args.neighbors], args.folds, args.seed, options)
    model = IDWModel(options).fit(dataset)
    if args.predict:
        report["point_prediction"] = asdict(model.predict_one(*args.predict))
    if args.grid_output:
        predictions = model.predict_grid(bounds, args.resolution)
        ReportWriter.write_predictions(args.grid_output, predictions)
        report["grid"] = {"output": args.grid_output, "points": len(predictions), "resolution": args.resolution}
    if args.report_output:
        ReportWriter.write_json(args.report_output, report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        ReportWriter.display(run(args))
        return 0
    except (ValueError, OSError, OverflowError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
