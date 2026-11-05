"""Phòng thí nghiệm AI – thiết kế thí nghiệm (A05, Research Lab).

Công cụ thống kê cơ bản, không cần thư viện ngoài:
- số mẫu cần cho mỗi nhóm (so sánh trung bình, so sánh tỷ lệ);
- chia lô ngẫu nhiên theo khối (RCBD) có hạt giống để tái lập được;
- kiểm định hoán vị (permutation test) cho hai nhóm — không giả định phân phối chuẩn;
- phiếu giả thuyết: ghi trước khi làm thí nghiệm, để không "chọn kết quả đẹp" sau đó.
Kết quả chỉ là gợi ý thiết kế; nhà khoa học thống kê duyệt trước khi làm thật.
"""

from __future__ import annotations

import math
import random
from datetime import datetime
from enum import Enum
from statistics import NormalDist, fmean

from pydantic import BaseModel, Field

_Z = NormalDist()


def _z(alpha: float, power: float, two_sided: bool) -> tuple[float, float]:
    if not (0 < alpha < 0.5 and 0.5 <= power < 1):
        raise ValueError("alpha trong (0; 0,5), power trong [0,5; 1)")
    za = _Z.inv_cdf(1 - alpha / 2) if two_sided else _Z.inv_cdf(1 - alpha)
    return za, _Z.inv_cdf(power)


def sample_size_means(delta: float, sd: float, alpha: float = 0.05, power: float = 0.8,
                      two_sided: bool = True) -> int:
    """Số đơn vị mỗi nhóm để phát hiện chênh lệch trung bình `delta` khi độ lệch chuẩn là `sd`."""
    if delta <= 0 or sd <= 0:
        raise ValueError("delta và sd phải dương")
    za, zb = _z(alpha, power, two_sided)
    return math.ceil(2 * ((za + zb) * sd / delta) ** 2)


def sample_size_proportions(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8,
                            two_sided: bool = True) -> int:
    """Số đơn vị mỗi nhóm để phân biệt hai tỷ lệ (ví dụ tỷ lệ sống 80% và 90%)."""
    if not (0 < p1 < 1 and 0 < p2 < 1) or p1 == p2:
        raise ValueError("p1, p2 trong (0; 1) và khác nhau")
    za, zb = _z(alpha, power, two_sided)
    pbar = (p1 + p2) / 2
    num = za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil((num / abs(p1 - p2)) ** 2)


class PlotAssignment(BaseModel):
    plot: str
    block: str
    treatment: str


def randomize_blocks(treatments: list[str], blocks: list[str], reps_per_block: int = 1,
                     seed: int = 0) -> list[PlotAssignment]:
    """Khối đầy đủ ngẫu nhiên: mỗi khối (ao, khu ruộng, dãy chuồng) có đủ mọi nghiệm thức."""
    if len(set(treatments)) != len(treatments) or len(treatments) < 2:
        raise ValueError("cần ít nhất 2 nghiệm thức khác nhau")
    if not blocks or reps_per_block < 1:
        raise ValueError("cần ít nhất 1 khối và 1 lần lặp")
    rng = random.Random(seed)
    out: list[PlotAssignment] = []
    for b in blocks:
        order = treatments * reps_per_block
        rng.shuffle(order)
        out.extend(PlotAssignment(plot=f"{b}-{i + 1}", block=b, treatment=t) for i, t in enumerate(order))
    return out


class TwoGroupResult(BaseModel):
    mean_a: float
    mean_b: float
    difference: float  # b − a
    p_value: float
    permutations: int


def permutation_test(a: list[float], b: list[float], permutations: int = 10_000, seed: int = 0) -> TwoGroupResult:
    if len(a) < 2 or len(b) < 2:
        raise ValueError("mỗi nhóm cần ít nhất 2 giá trị")
    rng = random.Random(seed)
    obs = fmean(b) - fmean(a)
    pooled = list(a) + list(b)
    na = len(a)
    extreme = 0
    for _ in range(permutations):
        rng.shuffle(pooled)
        if abs(fmean(pooled[na:]) - fmean(pooled[:na])) >= abs(obs) - 1e-12:
            extreme += 1
    return TwoGroupResult(mean_a=fmean(a), mean_b=fmean(b), difference=obs,
                          p_value=(extreme + 1) / (permutations + 1), permutations=permutations)


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"  # nhà khoa học duyệt thiết kế
    RUNNING = "running"
    CONCLUDED = "concluded"


class Hypothesis(BaseModel):
    id: str
    domain: str = Field(pattern=r"^D(0[1-9]|10)$")
    statement: str
    independent: str  # yếu tố thay đổi (ví dụ: liều vi sinh)
    dependent: str  # kết quả đo (ví dụ: tỷ lệ sống sau 30 ngày)
    controls: list[str] = Field(default_factory=list)
    expected_direction: str = Field(pattern=r"^(increase|decrease|no_change)$")
    primary_metric: str
    min_effect: float  # chênh lệch nhỏ nhất có ý nghĩa thực tế, ghi TRƯỚC khi làm
    alpha: float = 0.05
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    registered_at: datetime
    approved_by: str | None = None
    result: TwoGroupResult | None = None
    conclusion: str | None = None
