from enum import Enum
import random

class IrisClasification(Enum):
    SETOSA = 0
    VERSICOLOR = 1
    VIRGINICA = 2

def split_into_samples(data: list, start: int, take: int, features: int, label: int,
                       randomize: bool = False) -> tuple[list[float], list[int]]:
    if(randomize):
        random.shuffle(data)

    data = data[start:start+take]
    x: list[list[float]] = []
    y: list[int] = []
    for info in data:
        iris_info_compound = map(float, info[:features])
        x.append(list(iris_info_compound))
        y.append(_convert_label_to_calculable(info[label]))
    return x, y

def _convert_label_to_calculable(label: str) -> int:
    iris_switch = {
        "Iris-setosa": IrisClasification.SETOSA,
        "Iris-versicolor": IrisClasification.VERSICOLOR,
        "Iris-virginica": IrisClasification.VIRGINICA
    }
    return iris_switch[label].value