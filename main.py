import csv
import math
import random
import logging
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Iterator, Any, Union
from dataclasses import dataclass
import numpy as np
from scipy.spatial import KDTree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Finding:
    x: float
    y: float
    risk: float
    category: str


@dataclass
class Prediction:
    x: float
    y: float
    risk: float


class OwaspRiskMap:
    def __init__(self) -> None:
        self.findings: List[Finding] = []

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
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Line {row_num}: failed to parse - {e}")
                        continue
                        
            logger.info(f"Loaded {len(self.findings)} findings from {filename}")
            
        except FileNotFoundError:
            logger.error(f"File not found: {filename}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise

    def save_csv(self, filename: str, predictions: List[Prediction]) -> None:
        try:
            with open(filename, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y", "predicted_risk"])
                for pred in predictions:
                    writer.writerow([pred.x, pred.y, pred.risk])
            logger.info(f"Saved {len(predictions)} predictions to {filename}")
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")
            raise

    def create_grid(
        self, 
        xmin: float, 
        xmax: float, 
        ymin: float, 
        ymax: float, 
        resolution: int
    ) -> Iterator[Tuple[float, float]]:
        if xmin >= xmax:
            raise ValueError(f"xmin ({xmin}) must be less than xmax ({xmax})")
        if ymin >= ymax:
            raise ValueError(f"ymin ({ymin}) must be less than ymax ({ymax})")
        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")

        x_step = (xmax - xmin) / resolution
        y_step = (ymax - ymin) / resolution

        for i in range(resolution + 1):
            for j in range(resolution + 1):
                x = xmin + i * x_step
                y = ymin + j * y_step
                yield (x, y)

    def _validate_power(self, power: int) -> None:
        if power <= 0:
            raise ValueError(f"power must be positive, got {power}")

    def _normalize_max_points(self, max_points: Optional[int], total_points: int) -> Optional[int]:
        if max_points is not None and max_points <= 0:
            raise ValueError(f"max_points must be positive, got {max_points}")
        if max_points is None:
            return None
        if max_points > total_points:
            logger.warning(f"max_points ({max_points}) exceeds available points ({total_points}), using all points")
            return total_points
        return max_points

    def idw_risk(
        self,
        target_x: float,
        target_y: float,
        power: int = 2,
        max_points: Optional[int] = None,
        source: Optional[List[Finding]] = None
    ) -> float:
        self._validate_power(power)
        
        if source is None:
            source = self.findings

        if not source:
            logger.warning("No source points available for interpolation")
            return 0.0

        effective_max_points = self._normalize_max_points(max_points, len(source))

        distances_and_risks = []
        
        for finding in source:
            dist = math.hypot(finding.x - target_x, finding.y - target_y)
            if dist == 0.0:
                return finding.risk
            distances_and_risks.append((dist, finding.risk))

        if effective_max_points is not None and effective_max_points < len(distances_and_risks):
            distances_and_risks.sort(key=lambda item: item[0])
            distances_and_risks = distances_and_risks[:effective_max_points]

        weighted_sum = 0.0
        total_weight = 0.0

        for dist, risk in distances_and_risks:
            if dist > 0:
                weight = 1.0 / (dist ** power)
                weighted_sum += weight * risk
                total_weight += weight

        if total_weight == 0:
            return sum(r for _, r in distances_and_risks) / len(distances_and_risks)

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
        
        # Check for exact matches
        for dist, idx in zip(distances, indices):
            if dist == 0:
                idx_int = int(idx)
                if idx_int < len(findings):
                    return findings[idx_int].risk
                else:
                    logger.warning(f"Index {idx_int} out of range for findings size {len(findings)}")
                    return 0.0
            
        weighted_sum = 0.0
        total_weight = 0.0
        
        for dist, idx in zip(distances, indices):
            if dist > 0:
                weight = 1.0 / (dist ** power)
                idx_int = int(idx)
                if idx_int < len(findings):
                    weighted_sum += weight * findings[idx_int].risk
                    total_weight += weight
                
        if total_weight == 0:
            valid_risks = []
            for idx in indices:
                idx_int = int(idx)
                if idx_int < len(findings):
                    valid_risks.append(findings[idx_int].risk)
            if valid_risks:
                return sum(valid_risks) / len(valid_risks)
            return 0.0
            
        return weighted_sum / total_weight

    def _create_folds(self, data: List[Finding], k_folds: int) -> List[List[Finding]]:
        # Ensure k_folds doesn't exceed data size
        k_folds = min(k_folds, len(data))
        if k_folds == 0:
            return []
            
        fold_sizes = [len(data) // k_folds] * k_folds
        for i in range(len(data) % k_folds):
            fold_sizes[i] += 1
        
        folds = []
        start = 0
        for size in fold_sizes:
            folds.append(data[start:start + size])
            start += size
        return folds

    def category_summary(self) -> Dict[str, Dict[str, Union[int, float]]]:
        buckets: Dict[str, List[float]] = defaultdict(list)
        
        for finding in self.findings:
            buckets[finding.category].append(finding.risk)

        summary = {}
        for category, risks in buckets.items():
            if risks:
                summary[category] = {
                    "count": len(risks),
                    "mean_risk": sum(risks) / len(risks),
                    "min_risk": min(risks),
                    "max_risk": max(risks),
                }
        return summary

    def block_average(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        block_size: float = 10.0
    ) -> Dict[Tuple[float, float], float]:
        if xmin >= xmax:
            raise ValueError(f"xmin ({xmin}) must be less than xmax ({xmax})")
        if ymin >= ymax:
            raise ValueError(f"ymin ({ymin}) must be less than ymax ({ymax})")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        blocks: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        
        # Fixed: Calculate maximum block indices correctly
        max_bx = int((xmax - xmin) / block_size)
        max_by = int((ymax - ymin) / block_size)

        for finding in self.findings:
            if xmin <= finding.x <= xmax and ymin <= finding.y <= ymax:
                bx = int((finding.x - xmin) / block_size)
                by = int((finding.y - ymin) / block_size)
                # Clamp indices to valid range
                bx = min(bx, max_bx - 1) if bx >= max_bx else bx
                by = min(by, max_by - 1) if by >= max_by else by
                blocks[(bx, by)].append(finding.risk)

        averages = {}
        for (bx, by), risks in blocks.items():
            if risks:
                center_x = xmin + (bx + 0.5) * block_size
                center_y = ymin + (by + 0.5) * block_size
                averages[(center_x, center_y)] = sum(risks) / len(risks)

        return averages

    def statistics_summary(self) -> Dict[str, Any]:
        if not self.findings:
            return {}

        risks = [finding.risk for finding in self.findings]
        xs = [finding.x for finding in self.findings]
        ys = [finding.y for finding in self.findings]

        mean_risk = sum(risks) / len(risks)
        variance = sum((r - mean_risk) ** 2 for r in risks) / len(risks)

        return {
            "n_findings": len(self.findings),
            "mean_risk": mean_risk,
            "std_dev": math.sqrt(variance),
            "min_risk": min(risks),
            "max_risk": max(risks),
            "x_range": (min(xs), max(xs)),
            "y_range": (min(ys), max(ys)),
        }

    def cross_validate(
        self,
        power: int = 2,
        k_folds: int = 5,
        use_optimized: bool = False
    ) -> float:
        if not self.findings:
            logger.warning("No findings available for cross-validation")
            return 0.0
            
        if k_folds <= 0:
            raise ValueError(f"k_folds must be positive, got {k_folds}")

        self._validate_power(power)
        
        # Ensure we don't have more folds than data points
        k_folds = min(k_folds, len(self.findings))
        if k_folds < 2:
            logger.warning(f"Only {len(self.findings)} points available, using leave-one-out")
            k_folds = len(self.findings)
        
        shuffled = self.findings[:]
        random.shuffle(shuffled)
        
        folds = self._create_folds(shuffled, k_folds)
        errors = []

        for i, test_set in enumerate(folds):
            train_set = []
            for j, fold in enumerate(folds):
                if j != i:
                    train_set.extend(fold)
            
            if not train_set:
                logger.warning(f"Fold {i}: empty training set, skipping")
                continue
                
            if use_optimized:
                train_points = [(f.x, f.y) for f in train_set]
                tree = KDTree(train_points)
                
                for finding in test_set:
                    predicted = self._idw_risk_with_tree(
                        finding.x, finding.y, train_set, tree, power=power
                    )
                    errors.append((finding.risk - predicted) ** 2)
            else:
                for finding in test_set:
                    predicted = self.idw_risk(
                        finding.x, finding.y, power=power, source=train_set
                    )
                    errors.append((finding.risk - predicted) ** 2)

        rmse = math.sqrt(sum(errors) / len(errors)) if errors else 0.0
        logger.info(f"Cross-validation RMSE: {rmse:.3f}")
        return rmse

    def predict_grid(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        resolution: int,
        power: int = 2,
        max_points: Optional[int] = None,
        use_optimized: bool = False
    ) -> List[Prediction]:
        self._validate_power(power)
        
        if use_optimized and self.findings:
            train_points = [(f.x, f.y) for f in self.findings]
            tree = KDTree(train_points)
            
            predictions = []
            for x, y in self.create_grid(xmin, xmax, ymin, ymax, resolution):
                risk = self._idw_risk_with_tree(
                    x, y, self.findings, tree, power=power, max_points=max_points
                )
                predictions.append(Prediction(x, y, risk))
        else:
            predictions = []
            for x, y in self.create_grid(xmin, xmax, ymin, ymax, resolution):
                risk = self.idw_risk(x, y, power=power, max_points=max_points)
                predictions.append(Prediction(x, y, risk))
            
        logger.info(f"Generated {len(predictions)} predictions for grid {resolution}x{resolution}")
        return predictions

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
        if random_seed is not None:
            random.seed(random_seed)
            
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
            
        xmin, xmax = x_range
        ymin, ymax = y_range
        
        self.findings = []
        
        for _ in range(n_points):
            x = random.uniform(xmin, xmax)
            y = random.uniform(ymin, ymax)

            base = (
                0.03 * x +
                0.02 * y +
                math.sin(x / 15.0) * 0.8 +
                math.cos(y / 17.0) * 0.6
            )
            noise = random.gauss(0, noise_std)
            risk = max(0.0, min(10.0, risk_base + base + noise))

            category = random.choice(categories)
            self.findings.append(Finding(x, y, risk, category))
            
        logger.info(f"Generated {n_points} sample findings")


if __name__ == "__main__":
    tool = OwaspRiskMap()
    
    tool.generate_sample_data(n_points=250, random_seed=42)

    stats = tool.statistics_summary()
    print(f"Findings: {stats['n_findings']}")
    print(f"Mean risk: {stats['mean_risk']:.3f}")
    print(f"Std dev: {stats['std_dev']:.3f}")
    print(f"Risk range: {stats['min_risk']:.3f} -> {stats['max_risk']:.3f}")

    rmse = tool.cross_validate(power=2, k_folds=5)
    print(f"Cross-validation RMSE: {rmse:.3f}")
    
    rmse_optimized = tool.cross_validate(power=2, k_folds=5, use_optimized=True)
    print(f"Cross-validation RMSE (optimized): {rmse_optimized:.3f}")

    print("\nCategory summary:")
    for category, info in sorted(tool.category_summary().items()):
        print(
            f"{category:30s} "
            f"count={info['count']:3d} "
            f"mean={info['mean_risk']:.3f} "
            f"min={info['min_risk']:.3f} "
            f"max={info['max_risk']:.3f}"
        )

    predictions = tool.predict_grid(0, 100, 0, 100, resolution=20, power=2, max_points=12)
    tool.save_csv("owasp_risk_predictions.csv", predictions)
    
    predictions_optimized = tool.predict_grid(0, 100, 0, 100, resolution=20, power=2, max_points=12, use_optimized=True)
    tool.save_csv("owasp_risk_predictions_optimized.csv", predictions_optimized)

    print(f"\nSaved {len(predictions)} interpolated risk points to owasp_risk_predictions.csv")
    print(f"Saved {len(predictions_optimized)} interpolated risk points to owasp_risk_predictions_optimized.csv")
