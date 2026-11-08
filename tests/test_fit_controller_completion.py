"""Exercise the controller's completion block without importing the GPU stack."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize(
    "second_status,unchanged,expected",
    [
        ("RUNNING", False, {100: 0}),
        ("COMPLETED", False, {100: 0, 200: 2}),
        ("COMPLETED", True, {100: 0}),
    ],
    ids=["trailing-running", "multiple-completed", "trailing-unchanged"],
)
def test_completed_run_uses_its_own_shard(second_status, unchanged, expected):
    # Compile the actual scan and completion processing together, retaining the
    # variable scope that caused the bug. No controller initialization or GPUs.
    path = Path(__file__).resolve().parents[1] / "rapidfireai/fit/backend/controller.py"
    tree = ast.parse(path.read_text())
    polling_loop = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.While)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.operand, ast.Name)
        and node.test.operand.id == "all_done"
    )
    start = next(
        i for i, node in enumerate(polling_loop.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "completed_runs" for t in node.targets)
    )
    stop = next(
        i for i, node in enumerate(polling_loop.body)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name) and node.iter.id == "failed_runs"
    )
    block = ast.Module(body=polling_loop.body[start:stop], type_ignores=[])

    tasks = {
        0: dict(task_id=10, status="COMPLETED", run_id=100, shard_id=0),
        1: dict(task_id=20, status=second_status, run_id=200, shard_id=2),
    }
    scheduler = MagicMock()
    scheduler.run_ids = {100, 200}
    scheduler.state.run_visited_num_shards = {100: 1, 200: 3}
    controller = MagicMock()
    callback = MagicMock()
    callback.on_shard_complete.return_value = SimpleNamespace(action="continue")
    metrics = {"rougeL": 0.5}
    controller.metric_logger.get_run_metrics.return_value = metrics
    active_runs = {100: (0,), 200: (1,)}
    namespace = dict(
        self=controller,
        all_worker_tasks=tasks,
        prev_worker_tasks={1: tasks[1].copy()} if unchanged else {},
        all_run_details={
            run_id: dict(completed_steps=8, total_steps=128,
                         num_epochs_completed=0, metric_run_id=str(run_id))
            for run_id in active_runs
        },
        scheduler=scheduler,
        active_runs=active_runs,
        TaskStatus=SimpleNamespace(COMPLETED="COMPLETED", FAILED="FAILED"),
        shard_callback=callback,
        num_shards=4,
    )
    exec(compile(block, str(path), "exec"), namespace)

    calls = callback.on_shard_complete.call_args_list
    assert len(calls) == len(expected)
    assert {call.args[0]: call.args[1] for call in calls} == expected
    assert all(call.args[2] is metrics for call in calls)
    assert {call.args[0] for call in scheduler.set_completed_task.call_args_list} == set(expected)
    assert set(active_runs) == {100, 200} - set(expected)
