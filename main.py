import csv
import math
import random
import logging
import heapq
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Iterator
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class CategorySummaryEntry:
    category: str
    count: int
    mean_risk: float
    min_risk: float
    max_risk: float


@dataclass(frozen=True)
class StatisticsSummary:
    n_findings: int
    mean_risk: float
    std_dev: float
    min_risk: float
    max_risk: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class CrossValidationResult:
    rmse: float
    fold_count: int
    n_errors: int
    power: int
    optimized: bool


@dataclass(frozen=True)
class BlockAverageEntry:
    block_x: int
    block_y: int
    center_x: float
    center_y: float
    mean_risk: float
    count: int


@dataclass(frozen=True)
class GridDefinition:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    resolution: int


@dataclass(frozen=True)
class Bounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


@dataclass
class PredictorContext:
    findings: List[Finding]
    power: int
    max_points: Optional[int]
    use_optimized: bool
    tree: Optional[KDTree] = None


class IDWInterpolator:
    def __init__(self, power: int = 2, max_points: Optional[int] = None):
        self.power = power
        self.max_points = max_points
        self.tree: Optional[KDTree] = None
        self.values: Optional[np.ndarray] = None
        self.points: Optional[np.ndarray] = None

    def fit(self, points: np.ndarray, values: np.ndarray) -> None:
        if len(points) == 0:
            self.tree = None
            self.values = None
            self.points = None
            return
        self.tree = KDTree(points)
        self.values = values
        self.points = points

    def predict(self, targets: np.ndarray) -> np.ndarray:
        if self.tree is None or self.values is None or len(self.values) == 0:
            return np.zeros(len(targets))

        effective_max_points = self.max_points if self.max_points is not None else len(self.values)
        if effective_max_points > len(self.values):
            effective_max_points = len(self.values)

        k = effective_max_points
        distances, indices = self.tree.query(targets, k=k)

        if len(targets) == 1:
            distances = np.atleast_1d(distances)
            indices = np.atleast_1d(indices)

        predictions = np.zeros(len(targets))

        for i in range(len(targets)):
            if k == 1:
                dist = float(distances[i])
                idx = int(indices[i])
                if self._is_zero_distance(dist):
                    predictions[i] = float(self.values[idx])
                else:
                    dist_safe = max(dist, 1e-8)
                    weight = 1.0 / (dist_safe ** self.power)
                    predictions[i] = weight * float(self.values[idx]) / weight
            else:
                dists = distances[i]
                idxs = indices[i]
                valid_pairs = []
                for j in range(len(dists)):
                    dist = float(dists[j])
                    idx = int(idxs[j])
                    if self._is_zero_distance(dist):
                        predictions[i] = float(self.values[idx])
                        break
                    valid_pairs.append((dist, idx))
                else:
                    if not valid_pairs:
                        predictions[i] = 0.0
                        continue
                    weighted_sum = 0.0
                    total_weight = 0.0
                    for dist, idx in valid_pairs:
                        dist_safe = max(dist, 1e-8)
                        weight = 1.0 / (dist_safe ** self.power)
                        weighted_sum += weight * float(self.values[idx])
                        total_weight += weight
                    if total_weight == 0.0:
                        predictions[i] = sum(float(self.values[idx]) for _, idx in valid_pairs) / len(valid_pairs)
                    else:
                        predictions[i] = weighted_sum / total_weight

        return predictions

    def predict_single(self, target_x: float, target_y: float) -> float:
        return self.predict(np.array([[target_x, target_y]]))[0]

    def _is_zero_distance(self, dist: float, eps: float = 1e-12) -> bool:
        return abs(dist) <= eps


class OwaspRiskMap:
    def __init__(self) -> None:
        self.findings: List[Finding] = []

    def configure_logging(self, level: int = logging.INFO) -> None:
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(level=level)
        root_logger.setLevel(level)

    def load_csv(
        self,
        filename: str,
        x_col: int = 0,
        y_col: int = 1,
        risk_col: int = 2,
        category_col: int = 3,
        has_header: bool = True,
        reset: bool = False
    ) -> None:
        if reset:
            self.findings = []

        try:
            with open(filename, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                start_line = 1
                if has_header:
                    try:
                        next(reader)
                        start_line = 2
                    except StopIteration:
                        logger.warning("Empty CSV file")
                        return

                for row_num, row in enumerate(reader, start=start_line):
                    try:
                        if len(row) <= max(x_col, y_col, risk_col, category_col):
                            logger.warning(f"Line {row_num}: insufficient columns, skipping")
                            continue

                        x = float(row[x_col])
                        y = float(row[y_col])
                        risk = float(row[risk_col])
                        category = row[category_col].strip()

                        self.findings.append(Finding(x, y, risk, category))
                    except (ValueError, IndexError) as exc:
                        logger.warning(f"Line {row_num}: failed to parse - {exc}")
                        continue

            logger.info(f"Loaded {len(self.findings)} findings from {filename}")

        except FileNotFoundError:
            logger.error(f"File not found: {filename}")
            raise
        except Exception as exc:
            logger.error(f"Error loading CSV: {exc}")
            raise

    def save_csv(self, filename: str, predictions: List[Prediction]) -> None:
        try:
            with open(filename, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y", "predicted_risk"])
                for pred in predictions:
                    writer.writerow([pred.x, pred.y, pred.risk])
            logger.info(f"Saved {len(predictions)} predictions to {filename}")
        except Exception as exc:
            logger.error(f"Error saving CSV: {exc}")
            raise

    def _validate_bounds(self, xmin: float, xmax: float, ymin: float, ymax: float) -> Bounds:
        if xmin >= xmax:
            raise ValueError(f"xmin ({xmin}) must be less than xmax ({xmax})")
        if ymin >= ymax:
            raise ValueError(f"ymin ({ymin}) must be less than ymax ({ymax})")
        return Bounds(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    def _validate_resolution(self, resolution: int) -> int:
        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")
        return resolution

    def _validate_power(self, power: int) -> None:
        if power <= 0:
            raise ValueError(f"power must be positive, got {power}")

    def _normalize_max_points(self, max_points: Optional[int], total_points: int) -> Optional[int]:
        if max_points is not None and max_points <= 0:
            raise ValueError(f"max_points must be positive, got {max_points}")
        if max_points is None:
            return None
        if total_points <= 0:
            return None
        if max_points > total_points:
            logger.warning(f"max_points ({max_points}) exceeds available points ({total_points}), using all points")
            return total_points
        return max_points

    def create_grid(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        resolution: int
    ) -> Iterator[Tuple[float, float]]:
        bounds = self._validate_bounds(xmin, xmax, ymin, ymax)
        resolution = self._validate_resolution(resolution)

        x_step = (bounds.xmax - bounds.xmin) / resolution
        y_step = (bounds.ymax - bounds.ymin) / resolution

        for i in range(resolution + 1):
            for j in range(resolution + 1):
                x = bounds.xmin + i * x_step
                y = bounds.ymin + j * y_step
                yield (x, y)

    def _is_zero_distance(self, dist: float, eps: float = 1e-12) -> bool:
        return abs(dist) <= eps

    def _build_predictor_context(
        self,
        findings: List[Finding],
        power: int,
        max_points: Optional[int],
        use_optimized: bool
    ) -> PredictorContext:
        self._validate_power(power)

        if not findings:
            return PredictorContext(
                findings=[],
                power=power,
                max_points=None,
                use_optimized=False,
                tree=None,
            )

        effective_max_points = self._normalize_max_points(max_points, len(findings))
        tree = None

        if use_optimized:
            points = [(f.x, f.y) for f in findings]
            tree = KDTree(points)

        return PredictorContext(
            findings=findings,
            power=power,
            max_points=effective_max_points,
            use_optimized=use_optimized,
            tree=tree,
        )

    def _predict_with_context(self, context: PredictorContext, target_x: float, target_y: float) -> float:
        if not context.findings:
            logger.warning("No source points available for interpolation")
            return 0.0

        if context.use_optimized and context.tree is not None:
            return self._idw_risk_with_tree(
                target_x=target_x,
                target_y=target_y,
                findings=context.findings,
                tree=context.tree,
                power=context.power,
                max_points=context.max_points,
            )

        return self._idw_risk_naive(
            target_x=target_x,
            target_y=target_y,
            source=context.findings,
            power=context.power,
            max_points=context.max_points,
        )

    def _idw_risk_naive(
        self,
        target_x: float,
        target_y: float,
        source: List[Finding],
        power: int = 2,
        max_points: Optional[int] = None
    ) -> float:
        self._validate_power(power)

        if not source:
            logger.warning("No source points available for interpolation")
            return 0.0

        effective_max_points = self._normalize_max_points(max_points, len(source))
        distances_and_risks: List[Tuple[float, float]] = []

        for finding in source:
            dist = math.hypot(finding.x - target_x, finding.y - target_y)
            if self._is_zero_distance(dist):
                return finding.risk
            distances_and_risks.append((dist, finding.risk))

        if effective_max_points is not None and effective_max_points < len(distances_and_risks):
            distances_and_risks = heapq.nsmallest(effective_max_points, distances_and_risks, key=lambda item: item[0])

        weighted_sum = 0.0
        total_weight = 0.0

        for dist, risk in distances_and_risks:
            dist_safe = max(dist, 1e-8)
            weight = 1.0 / (dist_safe ** power)
            weighted_sum += weight * risk
            total_weight += weight

        if total_weight == 0.0:
            return sum(risk for _, risk in distances_and_risks) / len(distances_and_risks)

        return weighted_sum / total_weight

    def _idw_risk_with_tree(
        self,
        target_x: float,
        target_y: float,
        findings: List[Finding],
        tree: KDTree,
        power: int = 2,
        max_points: Optional[int] = None
    ) -> float:
        self._validate_power(power)

        if not findings:
            return 0.0

        effective_max_points = self._normalize_max_points(max_points, len(findings))
        k = len(findings) if effective_max_points is None else effective_max_points

        distances, indices = tree.query([target_x, target_y], k=k)
        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)

        valid_pairs: List[Tuple[float, int]] = []

        for dist, idx in zip(distances, indices):
            idx_int = int(idx)
            if 0 <= idx_int < len(findings):
                if self._is_zero_distance(float(dist)):
                    return findings[idx_int].risk
                valid_pairs.append((float(dist), idx_int))

        if not valid_pairs:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for dist, idx_int in valid_pairs:
            dist_safe = max(dist, 1e-8)
            weight = 1.0 / (dist_safe ** power)
            weighted_sum += weight * findings[idx_int].risk
            total_weight += weight

        if total_weight == 0.0:
            return sum(findings[idx_int].risk for _, idx_int in valid_pairs) / len(valid_pairs)

        return weighted_sum / total_weight

    def idw_risk(
        self,
        target_x: float,
        target_y: float,
        power: int = 2,
        max_points: Optional[int] = None,
        source: Optional[List[Finding]] = None,
        use_optimized: bool = False
    ) -> float:
        findings = self.findings if source is None else source
        context = self._build_predictor_context(
            findings=findings,
            power=power,
            max_points=max_points,
            use_optimized=use_optimized,
        )
        return self._predict_with_context(context, target_x, target_y)

    def _create_folds(self, data: List[Finding], k_folds: int) -> List[List[int]]:
        if not data:
            return []

        k_folds = min(k_folds, len(data))
        if k_folds <= 0:
            return []

        indices = list(range(len(data)))
        fold_sizes = [len(data) // k_folds] * k_folds
        for i in range(len(data) % k_folds):
            fold_sizes[i] += 1

        folds: List[List[int]] = []
        start = 0
        for size in fold_sizes:
            folds.append(indices[start:start + size])
            start += size
        return folds

    def category_summary(self) -> List[CategorySummaryEntry]:
        buckets: Dict[str, List[float]] = defaultdict(list)

        for finding in self.findings:
            buckets[finding.category].append(finding.risk)

        summary: List[CategorySummaryEntry] = []
        for category in sorted(buckets):
            risks = buckets[category]
            if risks:
                summary.append(
                    CategorySummaryEntry(
                        category=category,
                        count=len(risks),
                        mean_risk=sum(risks) / len(risks),
                        min_risk=min(risks),
                        max_risk=max(risks),
                    )
                )
        return summary

    def block_average(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        block_size: float = 10.0
    ) -> List[BlockAverageEntry]:
        bounds = self._validate_bounds(xmin, xmax, ymin, ymax)

        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        blocks: Dict[Tuple[int, int], List[float]] = defaultdict(list)

        block_count_x = max(1, math.ceil((bounds.xmax - bounds.xmin) / block_size))
        block_count_y = max(1, math.ceil((bounds.ymax - bounds.ymin) / block_size))

        for finding in self.findings:
            if bounds.xmin <= finding.x <= bounds.xmax and bounds.ymin <= finding.y <= bounds.ymax:
                bx = int((finding.x - bounds.xmin) / block_size)
                by = int((finding.y - bounds.ymin) / block_size)
                bx = min(max(bx, 0), block_count_x - 1)
                by = min(max(by, 0), block_count_y - 1)
                blocks[(bx, by)].append(finding.risk)

        results: List[BlockAverageEntry] = []
        for (bx, by), risks in sorted(blocks.items()):
            if risks:
                center_x = bounds.xmin + (bx + 0.5) * block_size
                center_y = bounds.ymin + (by + 0.5) * block_size
                results.append(
                    BlockAverageEntry(
                        block_x=bx,
                        block_y=by,
                        center_x=center_x,
                        center_y=center_y,
                        mean_risk=sum(risks) / len(risks),
                        count=len(risks),
                    )
                )

        return results

    def statistics_summary(self) -> Optional[StatisticsSummary]:
        if not self.findings:
            return None

        risks = [finding.risk for finding in self.findings]
        xs = [finding.x for finding in self.findings]
        ys = [finding.y for finding in self.findings]

        mean_risk = sum(risks) / len(risks)
        variance = sum((risk - mean_risk) ** 2 for risk in risks) / len(risks)

        return StatisticsSummary(
            n_findings=len(self.findings),
            mean_risk=mean_risk,
            std_dev=math.sqrt(variance),
            min_risk=min(risks),
            max_risk=max(risks),
            x_min=min(xs),
            x_max=max(xs),
            y_min=min(ys),
            y_max=max(ys),
        )

    def cross_validate(
        self,
        power: int = 2,
        k_folds: int = 5,
        use_optimized: bool = False,
        max_points: Optional[int] = None,
        random_seed: Optional[int] = None
    ) -> CrossValidationResult:
        if not self.findings:
            logger.warning("No findings available for cross-validation")
            return CrossValidationResult(
                rmse=0.0,
                fold_count=0,
                n_errors=0,
                power=power,
                optimized=use_optimized,
            )

        if k_folds <= 0:
            raise ValueError(f"k_folds must be positive, got {k_folds}")

        self._validate_power(power)

        effective_folds = min(k_folds, len(self.findings))
        if effective_folds < 2:
            logger.warning(f"Only {len(self.findings)} points available, using leave-one-out")
            effective_folds = len(self.findings)

        shuffled_indices = list(range(len(self.findings)))
        rng = random.Random(random_seed)
        rng.shuffle(shuffled_indices)

        folds = self._create_folds(shuffled_indices, effective_folds)
        squared_errors: List[float] = []

        for i, test_indices in enumerate(folds):
            train_indices = [idx for j, fold in enumerate(folds) if j != i for idx in fold]

            if not train_indices:
                logger.warning(f"Fold {i}: empty training set, skipping")
                continue

            train_findings = [self.findings[idx] for idx in train_indices]

            if use_optimized:
                points = np.array([[f.x, f.y] for f in train_findings])
                values = np.array([f.risk for f in train_findings])
                interpolator = IDWInterpolator(power=power, max_points=max_points)
                interpolator.fit(points, values)
                for test_idx in test_indices:
                    finding = self.findings[test_idx]
                    predicted = interpolator.predict_single(finding.x, finding.y)
                    squared_errors.append((finding.risk - predicted) ** 2)
            else:
                context = self._build_predictor_context(
                    findings=train_findings,
                    power=power,
                    max_points=max_points,
                    use_optimized=False,
                )
                for test_idx in test_indices:
                    finding = self.findings[test_idx]
                    predicted = self._predict_with_context(context, finding.x, finding.y)
                    squared_errors.append((finding.risk - predicted) ** 2)

        if not squared_errors:
            rmse = 0.0
        else:
            rmse = math.sqrt(sum(squared_errors) / len(squared_errors))

        logger.info(f"Cross-validation RMSE: {rmse:.3f}")

        return CrossValidationResult(
            rmse=rmse,
            fold_count=effective_folds,
            n_errors=len(squared_errors),
            power=power,
            optimized=use_optimized,
        )

    def optimize_power(self, k_folds: int = 5, use_optimized: bool = False, max_points: Optional[int] = None) -> int:
        powers = range(1, 6)
        best_power = 2
        best_rmse = float('inf')
        
        for power in powers:
            result = self.cross_validate(
                power=power,
                k_folds=k_folds,
                use_optimized=use_optimized,
                max_points=max_points
            )
            if result.rmse < best_rmse:
                best_rmse = result.rmse
                best_power = power
        
        logger.info(f"Optimal power selected: {best_power} (RMSE: {best_rmse:.3f})")
        return best_power

    def morans_i(self) -> Optional[float]:
        if len(self.findings) < 2:
            return None
        
        points = np.array([[f.x, f.y] for f in self.findings])
        values = np.array([f.risk for f in self.findings])
        
        tree = KDTree(points)
        n = len(values)
        mean_val = np.mean(values)
        
        numerator = 0.0
        denominator = 0.0
        
        for i in range(n):
            denominator += (values[i] - mean_val) ** 2
            distances, indices = tree.query(points[i], k=min(5, n))
            for j in range(1, len(distances)):
                if distances[j] > 0:
                    weight = 1.0 / distances[j]
                    numerator += weight * (values[i] - mean_val) * (values[indices[j]] - mean_val)
        
        if denominator == 0:
            return None
        
        return (n / sum(1.0 / d for d in distances if d > 0)) * (numerator / denominator)

    def predict_grid(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        resolution: int,
        power: int = 2,
        max_points: Optional[int] = None,
        use_optimized: bool = False,
        show_progress: bool = False
    ) -> List[Prediction]:
        bounds = self._validate_bounds(xmin, xmax, ymin, ymax)
        resolution = self._validate_resolution(resolution)

        if use_optimized and self.findings:
            points = np.array([[f.x, f.y] for f in self.findings])
            values = np.array([f.risk for f in self.findings])
            interpolator = IDWInterpolator(power=power, max_points=max_points)
            interpolator.fit(points, values)
            
            grid_points = list(self.create_grid(bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax, resolution))
            targets = np.array(grid_points)
            predictions_array = interpolator.predict(targets)
            
            predictions = [Prediction(x, y, float(risk)) for (x, y), risk in zip(grid_points, predictions_array)]
        else:
            context = self._build_predictor_context(
                findings=self.findings,
                power=power,
                max_points=max_points,
                use_optimized=False,
            )
            
            predictions: List[Prediction] = []
            iterator = self.create_grid(bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax, resolution)
            if show_progress:
                total_points = (resolution + 1) ** 2
                iterator = tqdm(iterator, total=total_points, desc="Generating predictions")
            
            for x, y in iterator:
                risk = self._predict_with_context(context, x, y)
                predictions.append(Prediction(x, y, risk))

        logger.info(f"Generated {len(predictions)} predictions for grid {resolution}x{resolution}")
        return predictions

    def predict_grid_lazy(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        resolution: int,
        power: int = 2,
        max_points: Optional[int] = None,
        use_optimized: bool = False
    ) -> Iterator[Prediction]:
        bounds = self._validate_bounds(xmin, xmax, ymin, ymax)
        resolution = self._validate_resolution(resolution)

        if use_optimized and self.findings:
            points = np.array([[f.x, f.y] for f in self.findings])
            values = np.array([f.risk for f in self.findings])
            interpolator = IDWInterpolator(power=power, max_points=max_points)
            interpolator.fit(points, values)
            
            for x, y in self.create_grid(bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax, resolution):
                risk = interpolator.predict_single(x, y)
                yield Prediction(x, y, risk)
        else:
            context = self._build_predictor_context(
                findings=self.findings,
                power=power,
                max_points=max_points,
                use_optimized=False,
            )
            
            for x, y in self.create_grid(bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax, resolution):
                risk = self._predict_with_context(context, x, y)
                yield Prediction(x, y, risk)

    def generate_sample_data(
        self,
        n_points: int = 250,
        x_range: Tuple[float, float] = (0, 100),
        y_range: Tuple[float, float] = (0, 100),
        risk_base: float = 4.5,
        noise_std: float = 0.35,
        categories: Optional[List[str]] = None,
        random_seed: Optional[int] = None
    ) -> None:
        if n_points <= 0:
            raise ValueError(f"n_points must be positive, got {n_points}")
        if noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")

        xmin, xmax = x_range
        ymin, ymax = y_range
        self._validate_bounds(xmin, xmax, ymin, ymax)

        rng = random.Random(random_seed)

        if categories is None:
            categories = [
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
            ]

        if not categories:
            raise ValueError("categories must not be empty")

        self.findings = []

        for _ in range(n_points):
            x = rng.uniform(xmin, xmax)
            y = rng.uniform(ymin, ymax)

            base = (
                0.03 * x +
                0.02 * y +
                math.sin(x / 15.0) * 0.8 +
                math.cos(y / 17.0) * 0.6
            )
            noise = rng.gauss(0, noise_std)
            risk = max(0.0, min(10.0, risk_base + base + noise))

            category = rng.choice(categories)
            self.findings.append(Finding(x, y, risk, category))

        logger.info(f"Generated {n_points} sample findings")


if __name__ == "__main__":
    tool = OwaspRiskMap()
    tool.configure_logging(logging.INFO)

    tool.generate_sample_data(n_points=250, random_seed=42)

    stats = tool.statistics_summary()
    if stats is not None:
        print(f"Findings: {stats.n_findings}")
        print(f"Mean risk: {stats.mean_risk:.3f}")
        print(f"Std dev: {stats.std_dev:.3f}")
        print(f"Risk range: {stats.min_risk:.3f} -> {stats.max_risk:.3f}")
        print(f"X range: {stats.x_min:.3f} -> {stats.x_max:.3f}")
        print(f"Y range: {stats.y_min:.3f} -> {stats.y_max:.3f}")

    moran = tool.morans_i()
    if moran is not None:
        print(f"Moran's I: {moran:.3f}")

    rmse = tool.cross_validate(power=2, k_folds=5, random_seed=42)
    print(f"Cross-validation RMSE: {rmse.rmse:.3f}")

    rmse_optimized = tool.cross_validate(power=2, k_folds=5, use_optimized=True, random_seed=42)
    print(f"Cross-validation RMSE (optimized): {rmse_optimized.rmse:.3f}")

    optimal_power = tool.optimize_power(k_folds=5, use_optimized=True)
    print(f"Optimal power: {optimal_power}")

    print("\nCategory summary:")
    for entry in tool.category_summary():
        print(
            f"{entry.category:30s} "
            f"count={entry.count:3d} "
            f"mean={entry.mean_risk:.3f} "
            f"min={entry.min_risk:.3f} "
            f"max={entry.max_risk:.3f}"
        )

    block_entries = tool.block_average(0, 100, 0, 100, block_size=20.0)
    print(f"\nBlock averages: {len(block_entries)} populated blocks")

    predictions = tool.predict_grid(0, 100, 0, 100, resolution=20, power=2, max_points=12, show_progress=True)
    tool.save_csv("owasp_risk_predictions.csv", predictions)

    predictions_optimized = tool.predict_grid(
        0,
        100,
        0,
        100,
        resolution=20,
        power=2,
        max_points=12,
        use_optimized=True,
        show_progress=True,
    )
    tool.save_csv("owasp_risk_predictions_optimized.csv", predictions_optimized)

    print(f"\nSaved {len(predictions)} interpolated risk points to owasp_risk_predictions.csv")
    print(f"Saved {len(predictions_optimized)} interpolated risk points to owasp_risk_predictions_optimized.csv")
