import csv
import math
import random
from collections import defaultdict

class OwaspRiskMap:
    def __init__(self):
        self.findings = []

    def load_csv(
        self,
        filename,
        x_col=0,
        y_col=1,
        risk_col=2,
        category_col=3,
        has_header=True,
        reset=False
    ):
        if reset:
            self.findings = []

        with open(filename, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            if has_header:
                try:
                    next(reader)
                except StopIteration:
                    return

            for row in reader:
                try:
                    x = float(row[x_col])
                    y = float(row[y_col])
                    risk = float(row[risk_col])
                    category = row[category_col].strip() if len(row) > category_col else "UNKNOWN"
                    self.findings.append((x, y, risk, category))
                except (ValueError, IndexError):
                    continue

    def save_csv(self, filename, predictions):
        with open(filename, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "predicted_risk"])
            for (x, y), risk in predictions:
                writer.writerow([x, y, risk])

    def create_grid(self, xmin, xmax, ymin, ymax, resolution):
        if resolution <= 0:
            raise ValueError("resolution must be positive")

        x_step = (xmax - xmin) / resolution
        y_step = (ymax - ymin) / resolution

        grid = []
        for i in range(resolution + 1):
            for j in range(resolution + 1):
                x = xmin + i * x_step
                y = ymin + j * y_step
                grid.append((x, y))
        return grid

    def idw_risk(self, target_x, target_y, power=2, max_points=None, source=None):
        if source is None:
            source = self.findings

        distances = []
        for x, y, risk, _category in source:
            dist = math.hypot(x - target_x, y - target_y)
            if dist == 0:
                return risk
            distances.append((dist, risk))

        if max_points:
            distances.sort(key=lambda item: item[0])
            distances = distances[:max_points]

        if not distances:
            if source:
                return sum(item[2] for item in source) / len(source)
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for dist, risk in distances:
            weight = 1 / (dist ** power)
            weighted_sum += weight * risk
            total_weight += weight

        return weighted_sum / total_weight if total_weight else 0.0

    def category_summary(self):
        buckets = defaultdict(list)
        for _x, _y, risk, category in self.findings:
            buckets[category].append(risk)

        summary = {}
        for category, risks in buckets.items():
            summary[category] = {
                "count": len(risks),
                "mean_risk": sum(risks) / len(risks),
                "min_risk": min(risks),
                "max_risk": max(risks),
            }
        return summary

    def block_average(self, xmin, xmax, ymin, ymax, block_size=10):
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        blocks = defaultdict(list)

        for x, y, risk, _category in self.findings:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                bx = int((x - xmin) / block_size)
                by = int((y - ymin) / block_size)
                blocks[(bx, by)].append(risk)

        averages = {}
        for (bx, by), risks in blocks.items():
            center_x = xmin + (bx + 0.5) * block_size
            center_y = ymin + (by + 0.5) * block_size
            averages[(center_x, center_y)] = sum(risks) / len(risks)

        return averages

    def statistics_summary(self):
        if not self.findings:
            return {}

        risks = [risk for _x, _y, risk, _category in self.findings]
        xs = [x for x, _y, _risk, _category in self.findings]
        ys = [y for _x, y, _risk, _category in self.findings]

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

    def cross_validate(self, power=2, k_folds=5):
        if not self.findings or k_folds <= 0:
            return 0.0

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

            for x, y, true_risk, _category in test_set:
                predicted = self.idw_risk(x, y, power=power, source=train_set)
                errors.append((true_risk - predicted) ** 2)

        return math.sqrt(sum(errors) / len(errors)) if errors else 0.0

    def predict_grid(self, xmin, xmax, ymin, ymax, resolution, power=2, max_points=None):
        grid = self.create_grid(xmin, xmax, ymin, ymax, resolution)
        predictions = []

        for x, y in grid:
            risk = self.idw_risk(x, y, power=power, max_points=max_points)
            predictions.append(((x, y), risk))

        return predictions


if __name__ == "__main__":
    tool = OwaspRiskMap()

    owasp_categories = [
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

    for _ in range(250):
        x = random.uniform(0, 100)   # e.g. component exposure
        y = random.uniform(0, 100)   # e.g. trust-boundary complexity

        base = (
            0.03 * x +
            0.02 * y +
            math.sin(x / 15.0) * 0.8 +
            math.cos(y / 17.0) * 0.6
        )
        noise = random.gauss(0, 0.35)
        risk = max(0.0, min(10.0, 4.5 + base + noise))

        category = random.choice(owasp_categories)
        tool.findings.append((x, y, risk, category))

    stats = tool.statistics_summary()
    print(f"Findings: {stats['n_findings']}")
    print(f"Mean risk: {stats['mean_risk']:.3f}")
    print(f"Std dev: {stats['std_dev']:.3f}")
    print(f"Risk range: {stats['min_risk']:.3f} -> {stats['max_risk']:.3f}")

    rmse = tool.cross_validate(power=2, k_folds=5)
    print(f"Cross-validation RMSE: {rmse:.3f}")

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

    print(f"\nSaved {len(predictions)} interpolated risk points to owasp_risk_predictions.csv")
