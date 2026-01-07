"""Custom observers used during optimisation runs."""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

from jmetal.core.observer import Observer
import csv


@dataclass
class _Record:
    """Container for observer data."""

    iteration: int
    values: Union[float, List[float]]

class ValueObserver(Observer):
    """
    逐次最良目的値を記録するObserver
    """
    def __init__(self) -> None:
        super().__init__()
        self._records: List[_Record] = []

    @property
    def iterations(self) -> List[int]:
        return [r.iteration for r in self._records]

    @property
    def best_objectives(self) -> List[Union[float, List[float]]]:
        return [r.values for r in self._records]

    def update(self, *args, **kwargs):
        evaluations = kwargs.get('EVALUATIONS')
        solutions = kwargs.get('SOLUTIONS')
        if solutions is None:
            return
        best = self._extract_best_values(solutions)
        self._records.append(_Record(evaluations, best))

    def _extract_best_values(self, solutions: Union[Sequence[Any], Any]) -> Union[float, List[float]]:
        """Return the best objective value(s) from the given solutions."""
        if isinstance(solutions, list):
            if hasattr(solutions[0], "objectives") and len(solutions[0].objectives) > 1:
                return [min(s.objectives[i] for s in solutions) for i in range(len(solutions[0].objectives))]
            return min(s.objectives[0] for s in solutions)

        if hasattr(solutions, "objectives") and len(solutions.objectives) > 1:
            return list(solutions.objectives)
        return solutions.objectives[0]

    def to_csv(self, path: str):
        """
        記録データをCSV保存
        """
        with open(path, mode="w", newline="") as f:
            writer = csv.writer(f)
            for rec in self._records:
                if isinstance(rec.values, list):
                    writer.writerow([rec.iteration] + rec.values)
                else:
                    writer.writerow([rec.iteration, rec.values])


class HyperVolumeObserver(Observer):
    """
    非支配解のハイパーボリュームを追跡
    """
    def __init__(self, ref_point: Optional[List[float]] = None):
        super().__init__()
        self.ref_point = ref_point
        self._records: List[_Record] = []

    @property
    def iterations(self) -> List[int]:
        return [r.iteration for r in self._records]

    @property
    def hypervolumes(self) -> List[float]:
        return [r.values for r in self._records]

    def update(self, *args, **kwargs):
        algorithm = kwargs.get('algorithm')
        if algorithm is None or not hasattr(algorithm, 'solutions'):
            return
        from jmetal.util.solution import get_non_dominated_solutions
        from jmetal.core.quality_indicator import HyperVolume
        solutions = algorithm.solutions
        if not solutions:
            return
        front = get_non_dominated_solutions(solutions)
        if self.ref_point is None:
            # 自動で参照点設定（適宜調整）
            obj_array = [s.objectives for s in front]
            self.ref_point = [max(x) * 1.1 for x in zip(*obj_array)]
        hv = HyperVolume(reference_point=self.ref_point)
        value = hv.compute([s.objectives for s in front])
        self._records.append(_Record(algorithm.evaluations, value))

    def to_csv(self, path: str):
        with open(path, mode="w", newline="") as f:
            writer = csv.writer(f)
            for rec in self._records:
                writer.writerow([rec.iteration, rec.values])


class SaveResultsObserver(Observer):
    """Observer that saves the results of another observer to CSV."""

    def __init__(self, csv_file: str, observer_instance: ValueObserver) -> None:
        super().__init__()
        self.csv_file = csv_file
        self.observer_instance = observer_instance

    def update(self, *args, **kwargs):  # noqa: D401
        """Write current observer data to ``csv_file``."""
        with open(self.csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            for iter_, values in zip(
                self.observer_instance.iterations,
                self.observer_instance.best_objectives,
            ):
                row = [iter_] + (values if isinstance(values, list) else [values])
                writer.writerow(row)

