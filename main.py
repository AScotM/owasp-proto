import csv
import math
import random
import logging
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Iterator, Any
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


class OwaspRiskMap:
    def __init__(self) -> None:
        self.findings: List[Finding] = []
        self._kdtree: Optional[KDTree] = None
        self._kdtree_points: Optional[List[Tuple[float, float]]] = None

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
            self._kdtree = None
            self._kdtree_points = None

        try:
            with open(filename, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                if has_header:
                    try:
                        next(reader)
                    except StopIteration:
                        logger.warning("Empty CSV file")
                        return

                for row_num, row in enumerate(reader, start=1 if has_header else 0):
                    try:
                        if len(row) <= max(x_col, y_col, risk_col, category_col):
                            logger.warning(f"Row {row_num}: insufficient columns, skipping")
                            continue
                            
                        x = float(row[x_col])
                        y = float(row[y_col])
                        risk = float(row[risk_col])
                        category = row[category_col].strip() if len(row) > category_col else "UNKNOWN"
                        self.findings.append(Finding(x, y, risk, category))
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Row {row_num}: failed to parse - {e}")
                        continue
                        
            self._build_kdtree()
            logger.info(f"Loaded {len(self.findings)} findings from {filename}")
            
        except FileNotFoundError:
            logger.error(f"File not found: {filename}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise

    def save_csv(self, filename: str, predictions: List[Tuple[Tuple[float, float], float]]) -> None:
        try:
            with open(filename, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y", "predicted_risk"])
                for (x, y), risk in predictions:
                    writer.writerow([x, y, risk])
            logger.info(f"Saved {len(predictions)} predictions to {filename}")
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")
            raise

    def _build_kdtree(self) -> None:
        if not self.findings:
            self._kdtree = None
            self._kdtree_points = None
            return
            
        points = [(f.x, f.y) for f in self.findings]
        self._kdtree_points = points
        self._kdtree = KDTree(points)

    def create_grid(self, xmin: float, xmax: float, ymin: float, ymax: float, resolution: int) -> Iterator[Tuple[float, float]]:
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

    def idw_risk(
        self,
        target_x: float,
        target_y: float,
        power: int = 2,
        max_points: Optional[int] = None,
        source: Optional[List[Finding]] = None
    ) -> float:
        if source is None:
            source = self.findings

        if not source:
            logger.warning("No source points available for interpolation")
            return 0.0

        if max_points is not None and max_points <= 0:
            raise ValueError(f"max_points must be positive, got {max_points}")

        distances_and_risks = []
        
        for finding in source:
            dist = math.hypot(finding.x - target_x, finding.y - target_y)
            if dist == 0.0:
                return finding.risk
            distances_and_risks.append((dist, finding.risk))

        if max_points is not None and max_points < len(distances_and_risks):
            distances_and_risks.sort(key=lambda item: item[0])
            distances_and_risks = distances_and_risks[:max_points]

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

    def idw_risk_optimized(
        self,
        target_x: float,
        target_y: float,
        power: int = 2,
        max_points: Optional[int] = None
    ) -> float:
        if not self.findings:
            return 0.0
            
        if self._kdtree is None or self._kdtree_points is None:
            self._build_kdtree()
            
        if max_points is None:
            max_points = len(self.findings)
            
        distances, indices = self._kdtree.query([target_x, target_y], k=min(max_points, len(self.findings)))
        
        if np.isscalar(distances):
            if distances == 0:
                return self.findings[indices].risk
            weight = 1.0 / (distances ** power)
            return weight * self.findings[indices].risk / weight
            
        weighted_sum = 0.0
        total_weight = 0.0
        
        for dist, idx in zip(distances, indices):
            if dist == 0:
                return self.findings[idx].risk
            if dist > 0:
                weight = 1.0 / (dist ** power)
                weighted_sum += weight * self.findings[idx].risk
                total_weight += weight
                
        if total_weight == 0:
            return sum(self.findings[idx].risk for idx in indices) / len(indices)
            
        return weighted_sum / total_weight

    def category_summary(self) -> Dict[str, Dict[str, Any]]:
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

        for finding in self.findings:
            if xmin <= finding.x <= xmax and ymin <= finding.y <= ymax:
                bx = int((finding.x - xmin) / block_size)
                by = int((finding.y - ymin) / block_size)
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

        k_folds = min(k_folds, len(self.findings))
        shuffled = self.findings[:]
        random.shuffle(shuffled)

        fold_size = max(1, len(shuffled) // k_folds)
        errors = []

        for fold in range(k_folds):
            start = fold * fold_size
            end = start + fold_size if fold < k_folds - 1 else len(shuffled)

            test_set = shuffled[start:end]
            train_set = shuffled[:start] + shuffled[end:]

            for finding in test_set:
                if use_optimized:
                    temp_tree = self._kdtree
                    temp_points = self._kdtree_points
                    self.findings = train_set
                    self._build_kdtree()
                    predicted = self.idw_risk_optimized(finding.x, finding.y, power=power)
                    self.findings = shuffled
                    self._kdtree = temp_tree
                    self._kdtree_points = temp_points
                else:
                    predicted = self.idw_risk(finding.x, finding.y, power=power, source=train_set)
                    
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
    ) -> List[Tuple[Tuple[float, float], float]]:
        predictions = []
        grid_points = list(self.create_grid(xmin, xmax, ymin, ymax, resolution))
        
        for x, y in grid_points:
            if use_optimized:
                risk = self.idw_risk_optimized(x, y, power=power, max_points=max_points)
            else:
                risk = self.idw_risk(x, y, power=power, max_points=max_points)
            predictions.append(((x, y), risk))
            
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
            
        self._build_kdtree()
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
