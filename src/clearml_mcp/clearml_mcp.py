"""ClearML MCP Server implementation."""

from typing import Any, Mapping

from clearml import Model, Task
from fastmcp import FastMCP

mcp = FastMCP("clearml-mcp")

VALID_SCALAR_X_AXIS = ("iter", "timestamp", "iso_time")


def _tool_error(message: str) -> dict[str, Any]:
    """Standard error payload for tools that use ok or error style."""
    return {"ok": False, "error": message}


def _truncate_xy(
    x: list[Any],
    y: list[Any],
    max_points: int | None,
) -> tuple[list[Any], list[Any]]:
    if max_points is None or max_points <= 0 or len(y) <= max_points:
        return x, y
    return x[-max_points:], y[-max_points:]


def _load_reported_scalars(
    task: object,
    *,
    full_series: bool,
    max_samples: int,
    x_axis: str,
) -> Mapping[str, Any]:
    if x_axis not in VALID_SCALAR_X_AXIS:
        msg = f"x_axis must be one of {VALID_SCALAR_X_AXIS}"
        raise ValueError(msg)
    if full_series:
        return task.get_all_reported_scalars(x_axis=x_axis)
    return task.get_reported_scalars(max_samples=max_samples, x_axis=x_axis)


def _variant_summary_and_series(
    data: dict[str, Any],
    *,
    include_series: bool,
    max_points_per_series: int | None,
) -> dict[str, Any]:
    ys = data.get("y") if data and "y" in data else None
    xs = data.get("x") if data and "x" in data else None
    if not ys:
        out: dict[str, Any] = {
            "last_value": None,
            "min_value": None,
            "max_value": None,
            "iterations": 0,
        }
        if include_series:
            out["x"] = []
            out["y"] = []
        return out

    out = {
        "last_value": ys[-1],
        "min_value": min(ys),
        "max_value": max(ys),
        "iterations": len(ys),
    }
    if include_series:
        xi = list(xs) if xs is not None else []
        yi = list(ys)
        xi, yi = _truncate_xy(xi, yi, max_points_per_series)
        out["x"] = xi
        out["y"] = yi
    return out


def _metrics_from_scalars(
    scalars: dict[str, Any],
    *,
    include_series: bool,
    max_points_per_series: int | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric, variants in scalars.items():
        metrics[metric] = {}
        for variant, data in variants.items():
            if not data or "y" not in data:
                continue
            metrics[metric][variant] = _variant_summary_and_series(
                data,
                include_series=include_series,
                max_points_per_series=max_points_per_series,
            )
    return metrics


def _merge_task_filter(
    status: str | None,
    page: int | None,
    page_size: int | None,
) -> dict[str, Any] | None:
    task_filter: dict[str, Any] = {}
    if status:
        task_filter["status"] = [status]
    if page_size is not None:
        task_filter["page"] = page if page is not None else 0
        task_filter["page_size"] = page_size
        task_filter.setdefault("order_by", ["-last_update"])
    return task_filter or None


def _script_to_dict(script: object) -> dict[str, Any]:
    if script is None:
        return {}
    keys = (
        "repository",
        "branch",
        "version_num",
        "entry_point",
        "working_dir",
        "diff",
        "binary",
        "requirements",
    )
    out: dict[str, Any] = {}
    for key in keys:
        if not hasattr(script, key):
            continue
        val = getattr(script, key, None)
        if key == "requirements" and val is not None:
            try:
                out[key] = dict(val)
            except (TypeError, ValueError):
                out[key] = str(val)
        else:
            out[key] = val
    return out


def initialize_clearml_connection() -> None:
    """Initialize and validate ClearML connection."""
    try:
        projects = Task.get_projects()
        if not projects:
            raise ValueError("No ClearML projects accessible - check your clearml.conf")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ClearML connection: {e!s}")


@mcp.tool()
async def get_connection_info() -> dict[str, Any]:
    """Return ClearML connectivity and server side project visibility (read only)."""
    try:
        projects = Task.get_projects()
        return {
            "ok": True,
            "accessible_project_count": len(projects),
        }
    except Exception as e:
        return _tool_error(f"Connection check failed: {e!s}")


@mcp.tool()
async def get_task_info(task_id: str) -> dict[str, Any]:
    """Get ClearML task details, parameters, and status."""
    try:
        task = Task.get_task(task_id=task_id)
        return {
            "id": task.id,
            "name": task.name,
            "status": task.status,
            "project": task.get_project_name(),
            "created": str(task.data.created),
            "last_update": str(task.data.last_update),
            "tags": list(task.data.tags) if task.data.tags else [],
            "type": task.task_type,
            "comment": task.comment if hasattr(task, "comment") else None,
        }
    except Exception as e:
        return {"error": f"Failed to get task info: {e!s}"}


@mcp.tool()
async def list_tasks(
    project_name: str | None = None,
    status: str | None = None,
    tags: list[str] | None = None,
    page: int = 0,
    page_size: int | None = None,
) -> list[dict[str, Any]]:
    """List ClearML tasks with filters. Optional page and page_size use server side paging."""
    try:
        task_filter = _merge_task_filter(status, page, page_size)
        # Task.query_tasks returns task IDs (strings), not task objects
        task_ids = Task.query_tasks(
            project_name=project_name,
            task_filter=task_filter,
            tags=tags,
        )

        # Convert task IDs to full task objects
        tasks = []
        for task_id in task_ids:
            try:
                task = Task.get_task(task_id=task_id)
                tasks.append(
                    {
                        "id": task.id,
                        "name": task.name,
                        "status": task.status,
                        "project": task.get_project_name(),
                        "created": str(task.data.created),
                        "tags": list(task.data.tags) if task.data.tags else [],
                    }
                )
            except Exception as e:
                # If we can't get a specific task, include the error but continue
                tasks.append({"id": task_id, "error": f"Failed to get task details: {e!s}"})

        return tasks
    except Exception as e:
        return [{"error": f"Failed to list tasks: {e!s}"}]


@mcp.tool()
async def get_task_parameters(task_id: str) -> dict[str, Any]:
    """Get task hyperparameters and configuration."""
    try:
        task = Task.get_task(task_id=task_id)
        return task.get_parameters_as_dict()
    except Exception as e:
        return {"error": f"Failed to get task parameters: {e!s}"}


@mcp.tool()
async def get_task_metrics(
    task_id: str,
    *,
    include_series: bool = False,
    full_series: bool = False,
    max_samples: int = 0,
    x_axis: str = "iter",
    max_points_per_series: int | None = None,
) -> dict[str, Any]:
    """Get task training metrics and scalars. Set include_series for x and y arrays."""
    try:
        task = Task.get_task(task_id=task_id)
        scalars = _load_reported_scalars(
            task,
            full_series=full_series,
            max_samples=max_samples,
            x_axis=x_axis,
        )
        return _metrics_from_scalars(
            scalars,
            include_series=include_series,
            max_points_per_series=max_points_per_series,
        )
    except Exception as e:
        return {"error": f"Failed to get task metrics: {e!s}"}


@mcp.tool()
async def get_task_artifacts(task_id: str) -> dict[str, Any]:
    """Get task artifacts and outputs."""
    try:
        task = Task.get_task(task_id=task_id)
        artifacts = task.artifacts

        artifact_dict = {}
        for key, artifact in artifacts.items():
            artifact_dict[key] = {
                "type": artifact.type,
                "mode": artifact.mode,
                "uri": artifact.uri,
                "content_type": artifact.content_type,
                "timestamp": str(artifact.timestamp) if hasattr(artifact, "timestamp") else None,
            }
        return artifact_dict
    except Exception as e:
        return {"error": f"Failed to get task artifacts: {e!s}"}


@mcp.tool()
async def get_task_code_provenance(task_id: str) -> dict[str, Any]:
    """Get repository, branch, commit, entry script, and diff captured for this task."""
    try:
        task = Task.get_task(task_id=task_id)
        script = getattr(task.data, "script", None) if task.data else None
        payload = _script_to_dict(script)
        return {"ok": True, "script": payload}
    except Exception as e:
        return _tool_error(f"Failed to get task code provenance: {e!s}")


@mcp.tool()
async def get_task_console_log(task_id: str, number_of_reports: int = 500) -> dict[str, Any]:
    """Get recent console log lines reported for this task."""
    try:
        task = Task.get_task(task_id=task_id)
        lines = task.get_reported_console_output(number_of_reports=number_of_reports)
        return {"ok": True, "lines": list(lines)}
    except Exception as e:
        return _tool_error(f"Failed to get task console log: {e!s}")


@mcp.tool()
async def get_task_configuration(
    task_id: str,
    section_name: str,
    *,
    as_dict: bool = False,
) -> dict[str, Any]:
    """Get a named configuration section blob from the task (for example Hydra or JSON config)."""
    try:
        task = Task.get_task(task_id=task_id)
        if as_dict:
            parsed = task.get_configuration_object_as_dict(name=section_name)
            return {"ok": True, "section_name": section_name, "value": parsed}
        text = task.get_configuration_object(name=section_name)
        return {"ok": True, "section_name": section_name, "value": text}
    except Exception as e:
        return _tool_error(f"Failed to get task configuration: {e!s}")


@mcp.tool()
async def get_model_info(task_id: str) -> dict[str, Any]:
    """Get model metadata and configuration."""
    try:
        task = Task.get_task(task_id=task_id)
        models = task.models

        model_info = {"input": [], "output": []}

        if models.get("input"):
            for model in models["input"]:
                model_info["input"].append(
                    {
                        "id": model.id,
                        "name": model.name,
                        "url": model.url,
                        "framework": model.framework,
                    },
                )

        if models.get("output"):
            for model in models["output"]:
                model_info["output"].append(
                    {
                        "id": model.id,
                        "name": model.name,
                        "url": model.url,
                        "framework": model.framework,
                    },
                )

        return model_info
    except Exception as e:
        return {"error": f"Failed to get model info: {e!s}"}


@mcp.tool()
async def list_models(project_name: str | None = None) -> list[dict[str, Any]]:
    """List available models with filtering."""
    try:
        models = Model.query_models(project_name=project_name)
        return [
            {
                "id": model.id,
                "name": model.name,
                "project": model.project,
                "framework": model.framework,
                "created": str(model.created),
                "tags": list(model.tags) if model.tags else [],
                "task_id": model.task,
            }
            for model in models
        ]
    except Exception as e:
        return [{"error": f"Failed to list models: {e!s}"}]


@mcp.tool()
async def get_model_artifacts(task_id: str) -> dict[str, Any]:
    """Get model files and download URLs."""
    try:
        task = Task.get_task(task_id=task_id)
        models = task.models

        artifacts = {"input_models": [], "output_models": []}

        if models.get("input"):
            for model in models["input"]:
                artifacts["input_models"].append(
                    {
                        "id": model.id,
                        "name": model.name,
                        "url": model.url,
                        "framework": model.framework,
                        "uri": model.uri,
                    },
                )

        if models.get("output"):
            for model in models["output"]:
                artifacts["output_models"].append(
                    {
                        "id": model.id,
                        "name": model.name,
                        "url": model.url,
                        "framework": model.framework,
                        "uri": model.uri,
                    },
                )

        return artifacts
    except Exception as e:
        return {"error": f"Failed to get model artifacts: {e!s}"}


@mcp.tool()
async def find_project_by_pattern(pattern: str) -> list[dict[str, Any]]:
    """Find ClearML projects by name pattern (case-insensitive)."""
    try:
        all_projects = Task.get_projects()
        matching_projects = []

        pattern_lower = pattern.lower()
        for proj in all_projects:
            if pattern_lower in proj.name.lower():
                matching_projects.append(
                    {
                        "id": getattr(proj, "id", None),
                        "name": proj.name,
                    }
                )

        return matching_projects
    except Exception as e:
        return [{"error": f"Failed to find projects by pattern: {e!s}"}]


@mcp.tool()
async def find_experiment_in_project(
    project_name: str, experiment_pattern: str
) -> list[dict[str, Any]]:
    """Find experiments in a specific project by name pattern."""
    try:
        # Get task IDs for the project
        task_ids = Task.query_tasks(project_name=project_name)

        matching_experiments = []
        pattern_lower = experiment_pattern.lower()

        for task_id in task_ids:
            try:
                task = Task.get_task(task_id=task_id)
                if pattern_lower in task.name.lower():
                    matching_experiments.append(
                        {
                            "id": task.id,
                            "name": task.name,
                            "status": task.status,
                            "project": task.get_project_name(),
                            "created": str(task.data.created),
                        }
                    )
            except Exception:
                # Skip tasks we can't access - could be permissions or API issues
                pass

        return matching_experiments
    except Exception as e:
        return [{"error": f"Failed to find experiments: {e!s}"}]


@mcp.tool()
async def list_projects() -> list[dict[str, Any]]:
    """List available ClearML projects."""
    try:
        projects = Task.get_projects()
        return [
            {
                "id": proj.id if hasattr(proj, "id") else None,
                "name": proj.name,
            }
            for proj in projects
        ]
    except Exception as e:
        return [{"error": f"Failed to list projects: {e!s}"}]


@mcp.tool()
async def get_project_stats(project_name: str) -> dict[str, Any]:
    """Get project statistics and task counts."""
    try:
        task_ids = Task.query_tasks(project_name=project_name)

        status_counts: dict[str, int] = {}
        task_types: set[str] = set()
        for task_id in task_ids:
            try:
                task = Task.get_task(task_id=task_id)
                st = task.status
                status_counts[st] = status_counts.get(st, 0) + 1
                tt = getattr(task, "task_type", None)
                if tt:
                    task_types.add(str(tt))
            except Exception:
                continue

        return {
            "project_name": project_name,
            "total_tasks": len(task_ids),
            "status_breakdown": status_counts,
            "task_types": sorted(task_types),
        }
    except Exception as e:
        return {"error": f"Failed to get project stats: {e!s}"}


@mcp.tool()
async def compare_tasks(
    task_ids: list[str],
    metrics: list[str] | None = None,
    *,
    include_series: bool = False,
    full_series: bool = False,
    max_samples: int = 0,
    x_axis: str = "iter",
    max_points_per_series: int | None = None,
) -> dict[str, Any]:
    """Compare multiple tasks by metrics."""
    try:
        comparison = {}

        for task_id in task_ids:
            task = Task.get_task(task_id=task_id)
            scalars = _load_reported_scalars(
                task,
                full_series=full_series,
                max_samples=max_samples,
                x_axis=x_axis,
            )

            task_metrics: dict[str, Any] = {"name": task.name, "status": task.status, "metrics": {}}

            if metrics:
                for metric in metrics:
                    if metric not in scalars:
                        continue
                    task_metrics["metrics"][metric] = _metrics_from_scalars(
                        {metric: scalars[metric]},
                        include_series=include_series,
                        max_points_per_series=max_points_per_series,
                    ).get(metric, {})
            else:
                task_metrics["metrics"] = _metrics_from_scalars(
                    scalars,
                    include_series=include_series,
                    max_points_per_series=max_points_per_series,
                )

            comparison[task_id] = task_metrics

        return comparison
    except Exception as e:
        return {"error": f"Failed to compare tasks: {e!s}"}


@mcp.tool()
async def search_tasks(
    query: str,
    project_name: str | None = None,
    page: int = 0,
    page_size: int | None = None,
) -> list[dict[str, Any]]:
    """Search tasks by name, tags, or description.
    Optional page and page_size slice the result list.
    """
    try:
        # Task.query_tasks returns task IDs (strings), not task objects
        task_ids = Task.query_tasks(project_name=project_name)

        matching_tasks = []
        query_lower = query.lower()

        for task_id in task_ids:
            try:
                task = Task.get_task(task_id=task_id)

                # Check if the task matches the search query
                task_name = task.name.lower()
                task_comment = getattr(task, "comment", "") or ""
                task_tags = list(task.data.tags) if task.data.tags else []

                if (
                    query_lower in task_name
                    or (task_comment and query_lower in task_comment.lower())
                    or any(query_lower in tag.lower() for tag in task_tags)
                ):
                    matching_tasks.append(
                        {
                            "id": task.id,
                            "name": task.name,
                            "status": task.status,
                            "project": task.get_project_name(),
                            "created": str(task.data.created),
                            "tags": task_tags,
                            "comment": task_comment,
                        }
                    )
            except Exception as e:
                # If we can't get a specific task, skip it but log the error
                matching_tasks.append(
                    {"id": task_id, "error": f"Failed to get task details: {e!s}"}
                )

        if page_size is not None:
            start = max(page, 0) * page_size
            end = start + page_size
            matching_tasks = matching_tasks[start:end]

        return matching_tasks
    except Exception as e:
        return [{"error": f"Failed to search tasks: {e!s}"}]


def main() -> None:
    """Entry point for uvx clearml-mcp."""
    initialize_clearml_connection()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
