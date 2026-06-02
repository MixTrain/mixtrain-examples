"""Launch a parameter sweep over another Mixtrain workflow."""

from decimal import Decimal
from typing import TypedDict

from mixtrain import MixFlow, Workflow


class SweepRun(TypedDict):
    value: int | float
    run_number: int
    status: str


class ParameterSweepOutput(TypedDict):
    runs: list[Workflow]
    sweep: list[SweepRun]
    count: int


class ParameterSweepWorkflow(MixFlow):
    """Run a numeric parameter sweep over another workflow.

    Tags: orchestration, sweep, workflow
    """

    def run(
        self,
        target_workflow: Workflow,
        sweep_param: str,
        start: float,
        stop: float,
        step: float,
        base_inputs: dict | None = None,
        sandbox: dict | None = None,
        max_runs: int = 100,
    ) -> ParameterSweepOutput:
        """Submit one target workflow run per generated sweep value.

        Args:
            target_workflow: Workflow to run for each sweep value
            sweep_param: Name of the target workflow input to vary
            start: First sweep value
            stop: Last sweep value, inclusive
            step: Increment between values
            base_inputs: Inputs to pass through to every child run
            sandbox: Optional sandbox overrides for every child run
            max_runs: Safety cap for the number of child runs
        """
        if step == 0:
            raise ValueError("step must be non-zero")
        if max_runs < 1:
            raise ValueError("max_runs must be at least 1")
        if step > 0 and start > stop:
            raise ValueError("start must be <= stop when step is positive")
        if step < 0 and start < stop:
            raise ValueError("start must be >= stop when step is negative")

        target_inputs = (target_workflow.metadata.get("spec") or {}).get("inputs") or {}
        if sweep_param not in target_inputs:
            valid_inputs = ", ".join(sorted(target_inputs)) or "none"
            raise ValueError(
                f"Target workflow '{target_workflow.name}' has no input "
                f"'{sweep_param}'. Valid inputs: {valid_inputs}"
            )

        sweep_type = target_inputs[sweep_param].get("type")
        numeric_types = {"int", "integer", "float", "number"}
        if sweep_type not in numeric_types:
            raise ValueError(
                f"Target input '{sweep_param}' must be numeric, got type '{sweep_type}'"
            )

        values = self._build_sweep_values(start, stop, step, sweep_type, max_runs)
        base_inputs = dict(base_inputs or {})
        sandbox = dict(sandbox or {})

        sweep: list[SweepRun] = []
        runs: list[Workflow] = []

        for value in values:
            inputs = {**base_inputs, sweep_param: value}
            run = target_workflow.submit(inputs=inputs, sandbox=sandbox)
            run_number = run["run_number"]

            print(
                f"Submitted {target_workflow.name} run #{run_number} "
                f"with {sweep_param}={value}"
            )

            sweep.append(
                {
                    "value": value,
                    "run_number": run_number,
                    "status": run.get("status", "pending"),
                }
            )
            runs.append(Workflow(target_workflow.name, run_number=run_number))

        return {"runs": runs, "sweep": sweep, "count": len(runs)}

    @staticmethod
    def _build_sweep_values(
        start: float,
        stop: float,
        step: float,
        sweep_type: str,
        max_runs: int,
    ) -> list[int | float]:
        current = Decimal(str(start))
        end = Decimal(str(stop))
        increment = Decimal(str(step))
        values: list[int | float] = []

        def in_range(value: Decimal) -> bool:
            return value <= end if increment > 0 else value >= end

        while in_range(current):
            if len(values) >= max_runs:
                raise ValueError(
                    f"Sweep would create more than max_runs={max_runs} child runs"
                )

            if sweep_type in {"int", "integer"}:
                if current != current.to_integral_value():
                    raise ValueError(
                        "Integer sweeps require whole-number start, stop, and step values"
                    )
                values.append(int(current))
            else:
                values.append(float(current))

            current += increment

        return values
