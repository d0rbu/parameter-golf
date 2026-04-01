"""
Per-step and kernel-level profiling utilities for parameter-golf training.

Enable via environment variable:
    PROFILE=1 python train_gpt.py ...

StepProfiler  -- wall-clock timing per training phase (data, compute, optimizer).
TrainingProfiler -- torch.profiler wrapper for kernel-level Chrome traces.
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass

import torch
from torch.profiler import ProfilerActivity


@dataclass
class StepTiming:
    """Wall-clock durations (seconds) for one training step."""

    data: float = 0.0
    compute: float = 0.0  # forward + backward combined
    optimizer: float = 0.0
    total: float = 0.0


class StepProfiler:
    """Tracks per-phase wall-clock timing within training steps.

    All timing uses ``torch.cuda.synchronize()`` + ``time.perf_counter()``
    so that GPU work is fully drained before reading the clock.

    The first *warmup_steps* steps are recorded but excluded from the
    printed summary so that JIT compilation and cache warm-up don't skew
    the statistics.

    Typical call order inside the training loop::

        prof.step_begin()
        # ... data loading ...
        prof.mark_data()
        for micro_step in range(grad_accum_steps):
            # forward + backward (interleaved)
            ...
        prof.mark_backward()
        # ... optimizer.step() ...
        prof.step_end()
    """

    def __init__(self, device: torch.device, warmup_steps: int = 3) -> None:
        self.device = device
        self.warmup_steps = warmup_steps
        self.timings: list[StepTiming] = []

        # Per-step scratch timestamps
        self._t_step: float = 0.0
        self._t_data: float = 0.0
        self._t_backward: float = 0.0

    # ------------------------------------------------------------------
    # Markers (called from the training loop)
    # ------------------------------------------------------------------

    def step_begin(self) -> None:
        torch.cuda.synchronize(self.device)
        self._t_step = time.perf_counter()

    def mark_data(self) -> None:
        torch.cuda.synchronize(self.device)
        self._t_data = time.perf_counter()

    def mark_backward(self) -> None:
        torch.cuda.synchronize(self.device)
        self._t_backward = time.perf_counter()

    def step_end(self) -> None:
        """Marks the end of the optimizer phase and records the step."""
        torch.cuda.synchronize(self.device)
        t_end = time.perf_counter()

        timing = StepTiming(
            data=self._t_data - self._t_step,
            compute=self._t_backward - self._t_data,
            optimizer=t_end - self._t_backward,
            total=t_end - self._t_step,
        )
        self.timings.append(timing)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a table with Median, Mean, Std, Min, Max per phase.

        Warmup steps are excluded.  Median is shown first because it is
        the most representative metric for timing data.
        """
        warm = self.timings[self.warmup_steps :]
        if not warm:
            print("[StepProfiler] Not enough steps recorded after warmup.")
            return

        phases = ["Data", "Compute", "Optimizer", "Total"]
        keys = ["data", "compute", "optimizer", "total"]

        def _ms(secs: float) -> float:
            return secs * 1000.0

        print(f"\n{'Phase':<12} {'Median':>10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}  (ms)")
        print("-" * 72)

        for phase_name, key in zip(phases, keys):
            vals = [getattr(t, key) for t in warm]
            med = _ms(statistics.median(vals))
            avg = _ms(statistics.mean(vals))
            std = _ms(statistics.stdev(vals)) if len(vals) > 1 else 0.0
            lo = _ms(min(vals))
            hi = _ms(max(vals))
            print(f"{phase_name:<12} {med:>10.2f} {avg:>10.2f} {std:>10.2f} {lo:>10.2f} {hi:>10.2f}")

        print(f"\n({len(warm)} steps after {self.warmup_steps} warmup)")


class TrainingProfiler:
    """Wrapper around ``torch.profiler.profile`` for kernel-level tracing.

    Produces Chrome-compatible trace JSON and a kernel summary table.

    Usage::

        tp = TrainingProfiler(run_id="my_run")
        tp.start()
        for step in range(total_steps):
            # ... training ...
            tp.step()
        tp.stop()
        tp.export_trace("traces/my_run.json")
        tp.print_kernel_summary()
    """

    def __init__(
        self,
        run_id: str,
        active_steps: int = 10,
        wait_steps: int = 3,
        warmup_steps: int = 2,
    ) -> None:
        self.run_id = run_id
        self._schedule = torch.profiler.schedule(
            wait=wait_steps,
            warmup=warmup_steps,
            active=active_steps,
            repeat=1,
        )
        self._profiler: torch.profiler.profile | None = None

    def start(self) -> None:
        self._profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=self._schedule,
            with_stack=True,
            record_shapes=True,
        )
        self._profiler.__enter__()

    def step(self) -> None:
        if self._profiler is not None:
            self._profiler.step()

    def stop(self) -> None:
        if self._profiler is not None:
            self._profiler.__exit__(None, None, None)

    def export_trace(self, path: str) -> None:
        """Write a Chrome trace JSON to *path*, creating directories as needed."""
        if self._profiler is None:
            print("[TrainingProfiler] No profiler data to export.")
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._profiler.export_chrome_trace(path)
        print(f"[TrainingProfiler] Trace exported to {path}")

    def print_kernel_summary(self, top_n: int = 20) -> None:
        """Print the top *top_n* CUDA kernels sorted by total CUDA time."""
        if self._profiler is None:
            print("[TrainingProfiler] No profiler data to summarize.")
            return
        print(
            self._profiler.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=top_n,
            )
        )
