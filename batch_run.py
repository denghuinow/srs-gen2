#!/usr/bin/env python3
"""srs-gen2 批处理脚本"""
import argparse
import csv
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from config import get_config


PROJECT_ROOT = Path(__file__).resolve().parent
MAIN_FILE = PROJECT_ROOT / "main.py"


@dataclass(frozen=True)
class TaskInput:
    """单个任务的输入文件集合"""

    name: str
    d_orig: Path
    r_base: Optional[Path]
    d_base: Optional[Path]


@dataclass
class TaskResult:
    """任务执行结果"""

    name: str
    success: bool
    message: str
    elapsed: float
    skipped: bool = False
    stage_timings: Optional[Dict[str, float]] = None


def normalize_extensions(extensions: Sequence[str]) -> List[str]:
    """规范化扩展名，确保以点开头"""
    normalized = []
    for ext in extensions:
        if not ext:
            continue
        if not ext.startswith("."):
            normalized.append(f".{ext.lower()}")
        else:
            normalized.append(ext.lower())
    return list(dict.fromkeys(normalized))


def resolve_input_file(directory: Path, file_name: str, extensions: Sequence[str]) -> Optional[Path]:
    """根据文件名在目录中查找文件"""
    direct_path = (directory / file_name).resolve()
    if direct_path.exists():
        return direct_path
    stem = Path(file_name).stem
    for ext in extensions:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate.resolve()
    # 尝试模糊匹配
    lower_stem = stem.lower()
    for file in directory.iterdir():
        if not file.is_file():
            continue
        if file.suffix.lower() not in extensions:
            continue
        file_stem = file.stem.lower()
        if lower_stem == file_stem or lower_stem in file_stem or file_stem in lower_stem:
            return file.resolve()
    return None


def match_file_by_stem(stem: str, directory: Path, extensions: Sequence[str]) -> Optional[Path]:
    """在目录中查找与 stem 匹配的文件"""
    lower_stem = stem.lower()
    for ext in extensions:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate.resolve()
        # 再尝试大小写不同但扩展名相同
        candidate = directory / f"{stem}{ext.upper()}"
        if candidate.exists():
            return candidate.resolve()
    for file in directory.iterdir():
        if not file.is_file():
            continue
        if file.suffix.lower() not in extensions:
            continue
        file_stem = file.stem.lower()
        if file_stem == lower_stem or lower_stem in file_stem or file_stem in lower_stem:
            return file.resolve()
    return None


def copy_task_documents(tasks: Sequence[TaskInput], attr_name: str, destination: Path) -> int:
    """将任务对应的参考文档复制到指定目录，仅复制匹配到的文件"""

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    copied = 0
    for task in tasks:
        source = getattr(task, attr_name, None)
        if not source:
            continue
        shutil.copy2(source, destination / source.name)
        copied += 1
    return copied


def find_matching_files(
    d_orig_dir: Path,
    r_base_dir: Path,
    d_base_dir: Path,
    extensions: Sequence[str],
    specified_files: Optional[Sequence[str]] = None,
) -> List[TaskInput]:
    """匹配三个目录中的同名文件"""
    extensions = normalize_extensions(extensions)
    d_orig_files: List[Path] = []

    if specified_files:
        for file_name in specified_files:
            resolved = resolve_input_file(d_orig_dir, file_name, extensions)
            if resolved:
                d_orig_files.append(resolved)
            else:
                print(f"警告：未在 {d_orig_dir} 中找到指定文件 {file_name}", file=sys.stderr)
    else:
        for ext in extensions:
            d_orig_files.extend(d_orig_dir.glob(f"*{ext}"))

    if not d_orig_files:
        return []

    # 去重并按文件名排序
    unique_files = {file.resolve(): file for file in d_orig_files}
    sorted_files = sorted(unique_files.keys(), key=lambda p: p.stem.lower())

    tasks: List[TaskInput] = []
    for d_orig in sorted_files:
        task_name = d_orig.stem
        r_base = match_file_by_stem(task_name, r_base_dir, extensions)
        if not r_base:
            print(f"警告：未找到 {task_name} 的 r-base 文件", file=sys.stderr)
        d_base = match_file_by_stem(task_name, d_base_dir, extensions)
        if not d_base:
            print(f"警告：未找到 {task_name} 的 d-base 文件", file=sys.stderr)
        tasks.append(TaskInput(name=task_name, d_orig=d_orig, r_base=r_base, d_base=d_base))
    return tasks


def is_retryable_error(error_msg: str) -> bool:
    """判断错误是否可重试"""
    patterns = [
        "connection reset",
        "connection refused",
        "connection aborted",
        "connection timeout",
        "timeout",
        "network is unreachable",
        "no route to host",
        "operation not permitted",
        "connect error",
        "temporary failure",
        "service temporarily unavailable",
        "too many requests",
        "rate limit",
        "429",
        "502",
        "503",
        "504",
        "max_tokens",
        "max_completion_tokens",
    ]
    lower_msg = error_msg.lower()
    return any(pattern in lower_msg for pattern in patterns)


def filter_error_message(error_msg: str) -> str:
    """过滤掉噪声警告，保留核心错误"""
    error_lines = error_msg.splitlines()
    filtered: List[str] = []
    skip_warning = False
    for line in error_lines:
        if any(keyword in line for keyword in ["PyTorch", "TensorFlow", "Flax", "Models won't be available"]):
            skip_warning = True
            continue
        if skip_warning and any(token in line for token in ["Error", "error", "失败", "Exception"]):
            skip_warning = False
        if not skip_warning:
            filtered.append(line)
    merged = "\n".join(l for l in filtered if l.strip())
    target = merged if merged else "\n".join(error_lines[-20:])
    target_lines = [line for line in target.splitlines() if line.strip()]
    max_lines = 40
    if len(target_lines) > max_lines:
        target_lines = target_lines[-max_lines:]
    return "\n".join(target_lines).strip()


def run_subprocess_with_logging(
    cmd: Sequence[str],
    output_dir: Path,
    attempt_index: int,
) -> Tuple[subprocess.CompletedProcess, Path]:
    """以实时日志方式运行子进程"""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"run_attempt_{attempt_index + 1}.log"
    record_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stdout_buffer: List[str] = []
    stderr_buffer: List[str] = []

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"时间：{record_time}\n")
        log_file.write(f"命令：{' '.join(cmd)}\n")
        log_file.write("\n--- 实时日志开始 ---\n")
        log_file.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        log_lock = threading.Lock()

        def _pump(stream, buffer: List[str], label: str) -> None:
            try:
                for line in iter(stream.readline, ""):
                    buffer.append(line)
                    with log_lock:
                        log_file.write(f"[{label}] {line}")
                        log_file.flush()
            finally:
                stream.close()

        stdout_thread = threading.Thread(
            target=_pump,
            args=(process.stdout, stdout_buffer, "STDOUT"),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_pump,
            args=(process.stderr, stderr_buffer, "STDERR"),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        process.wait()
        stdout_thread.join()
        stderr_thread.join()
        log_file.write(f"\n--- 子进程退出：{process.returncode} ---\n")
        log_file.flush()

    latest_log = output_dir / "run_latest.log"
    try:
        shutil.copyfile(log_path, latest_log)
    except Exception:
        pass

    completed = subprocess.CompletedProcess(
        cmd,
        process.returncode,
        "".join(stdout_buffer),
        "".join(stderr_buffer),
    )
    return completed, log_path


def format_log_reference(log_path: Path, base_dir: Path) -> str:
    """返回适合展示的日志路径"""
    try:
        return str(log_path.relative_to(base_dir))
    except ValueError:
        return str(log_path)


def is_task_completed(output_dir: Path, expected_iterations: Optional[int] = None) -> bool:
    """检查任务输出目录是否已有完整结果"""
    if not output_dir.exists():
        return False
    required_files = ["srs_no-explore-clarify.md", "srs_no-clarify.md"]
    for file_name in required_files:
        if not (output_dir / file_name).exists():
            return False
    iter_files = sorted(output_dir.glob("srs_iter_*.md"))
    if not iter_files:
        return False
    if expected_iterations:
        for idx in range(1, expected_iterations + 1):
            if not (output_dir / f"srs_iter_{idx}.md").exists():
                return False
    return True


def collect_generated_versions(output_dir: Path) -> List[str]:
    """提取已生成的版本信息"""
    versions: List[str] = []
    for special in ["srs_no-explore-clarify", "srs_no-clarify"]:
        if (output_dir / f"{special}.md").exists():
            versions.append(special)
    for srs_file in output_dir.glob("srs_iter_*.md"):
        versions.append(srs_file.stem)

    def version_key(name: str) -> Tuple[int, int]:
        if name.startswith("srs_iter_"):
            try:
                return (1, int(name.replace("srs_iter_", "")))
            except ValueError:
                return (1, 0)
        if name == "srs_no-explore-clarify":
            return (0, 0)
        if name == "srs_no-clarify":
            return (0, 1)
        return (2, 0)

    return [name for name in sorted(set(versions), key=version_key)]


def _parse_iteration_index(name: str) -> Optional[int]:
    """解析 srs_iter_* 文件名中的迭代编号"""
    if not name.startswith("srs_iter_"):
        return None
    try:
        return int(name.replace("srs_iter_", ""))
    except ValueError:
        return None


def collect_stage_timings(output_dir: Path, run_start_time: float) -> Dict[str, float]:
    """根据输出目录中的文件生成阶段耗时（相对任务开始时间）"""

    def _duration_from(path: Path) -> Optional[float]:
        if not path.exists():
            return None
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return None
        duration = mtime - run_start_time
        if duration < 0:
            return 0.0
        return duration

    stage_files = [
        ("no-explore-clarify", output_dir / "srs_no-explore-clarify.md"),
        ("no-clarify", output_dir / "srs_no-clarify.md"),
    ]
    timings: Dict[str, float] = {}
    for stage_name, stage_path in stage_files:
        duration = _duration_from(stage_path)
        if duration is not None:
            timings[stage_name] = duration

    iter_files: List[Tuple[int, Path]] = []
    for srs_file in output_dir.glob("srs_iter_*.md"):
        iter_idx = _parse_iteration_index(srs_file.stem)
        if iter_idx is None:
            continue
        iter_files.append((iter_idx, srs_file))

    for iter_idx, srs_file in sorted(iter_files, key=lambda item: item[0]):
        duration = _duration_from(srs_file)
        if duration is not None:
            timings[f"iter{iter_idx}"] = duration

    return timings


def run_single_task(
    task: TaskInput,
    output_base_dir: Path,
    rho: Optional[float],
    max_outer_iter: Optional[int],
    max_inner_iter: Optional[int],
    max_retries: int,
    retry_delay: float,
    skip_existing: bool,
    expected_iterations: Optional[int],
) -> TaskResult:
    """执行单个任务，包含重试逻辑"""
    output_dir = output_base_dir / task.name
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_inputs = []
    if task.r_base is None:
        missing_inputs.append("r-base")
    if task.d_base is None:
        missing_inputs.append("d-base")
    if missing_inputs:
        msg = f"缺少输入文件：{', '.join(missing_inputs)}"
        return TaskResult(name=task.name, success=False, message=msg, elapsed=0.0)

    if skip_existing and is_task_completed(output_dir, expected_iterations):
        return TaskResult(
            name=task.name,
            success=True,
            message="已跳过（检测到完整输出）",
            elapsed=0.0,
            skipped=True,
        )

    cmd = [
        sys.executable,
        str(MAIN_FILE),
        "--d-orig",
        str(task.d_orig),
        "--r-base",
        str(task.r_base),
        "--d-base",
        str(task.d_base),
        "--output-dir",
        str(output_dir),
    ]

    if rho is not None:
        cmd.extend(["--rho", str(rho)])
    if max_outer_iter is not None:
        cmd.extend(["--max-outer-iter", str(max_outer_iter)])
    if max_inner_iter is not None:
        cmd.extend(["--max-inner-iter", str(max_inner_iter)])

    last_error = ""
    last_log_path: Optional[Path] = None
    start_time = time.time()
    for attempt in range(max_retries + 1):
        try:
            completed, log_path = run_subprocess_with_logging(
                cmd=cmd,
                output_dir=output_dir,
                attempt_index=attempt,
            )
            last_log_path = log_path
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(
                    completed.returncode,
                    cmd,
                    output=completed.stdout,
                    stderr=completed.stderr,
                )
            log_ref = format_log_reference(log_path, output_base_dir)
            elapsed = time.time() - start_time
            versions = collect_generated_versions(output_dir)
            version_info = f"，版本：{', '.join(versions)}" if versions else ""
            extra = f"，重试 {attempt} 次" if attempt > 0 else ""
            message = f"成功（耗时 {elapsed:.2f} 秒{extra}{version_info}，日志：{log_ref}）"
            stage_timings = collect_stage_timings(output_dir, start_time)
            return TaskResult(
                name=task.name,
                success=True,
                message=message,
                elapsed=elapsed,
                stage_timings=stage_timings,
            )
        except subprocess.CalledProcessError as exc:
            elapsed = time.time() - start_time
            raw_error = exc.stderr or exc.stdout or str(exc)
            log_ref = ""
            if last_log_path:
                log_ref = format_log_reference(last_log_path, output_base_dir)
            filtered = filter_error_message(raw_error)
            last_error = filtered
            if attempt < max_retries and is_retryable_error(filtered):
                time.sleep(retry_delay)
                continue
            if len(filtered) > 1000:
                filtered = f"{filtered[:1000]}..."
            extra = f"，重试 {attempt} 次" if attempt > 0 else ""
            if log_ref:
                log_line = f"\n日志：{log_ref}"
            else:
                log_line = ""
            message = f"失败（耗时 {elapsed:.2f} 秒{extra}）：{filtered}{log_line}"
            return TaskResult(name=task.name, success=False, message=message, elapsed=elapsed)

    elapsed = time.time() - start_time
    final_msg = last_error or "未知错误"
    if last_log_path:
        log_ref = format_log_reference(last_log_path, output_base_dir)
        final_msg = f"{final_msg}\n日志：{log_ref}"
    if len(final_msg) > 1000:
        final_msg = f"{final_msg[:1000]}..."
    return TaskResult(
        name=task.name,
        success=False,
        message=f"失败（耗时 {elapsed:.2f} 秒）：{final_msg}",
        elapsed=elapsed,
    )


def apply_startup_delay(base_delay: float, growth: float, index: int) -> None:
    """根据配置在提交任务前休眠"""
    if base_delay <= 0 or index <= 0:
        return
    factor = 1.0 + max(growth, 0.0) * index if growth > 0 else 1.0
    time.sleep(base_delay * factor)


def collect_outputs(
    output_base_dir: Path,
    task_map: Dict[str, TaskInput],
    results: Sequence[TaskResult],
) -> Tuple[int, int]:
    """收集所有成功任务的输出"""
    srs_collection_dir = output_base_dir / "srs_collection"
    units_collection_dir = output_base_dir / "units_collection"
    srs_collection_dir.mkdir(exist_ok=True)
    units_collection_dir.mkdir(exist_ok=True)

    srs_count = 0
    units_count = 0

    for result in results:
        if not result.success:
            continue
        task = task_map.get(result.name)
        if not task:
            continue
        task_output_dir = output_base_dir / task.name
        if not task_output_dir.exists():
            continue

        # 收集 SRS 文档
        for srs_file in task_output_dir.glob("srs_iter_*.md"):
            version_dir = srs_collection_dir / srs_file.stem
            version_dir.mkdir(parents=True, exist_ok=True)
            dest = version_dir / f"{task.name}{srs_file.suffix}"
            shutil.copy2(srs_file, dest)
            srs_count += 1
        for special in ["srs_no-explore-clarify.md", "srs_no-clarify.md"]:
            source = task_output_dir / special
            if source.exists():
                version_dir = srs_collection_dir / source.stem
                version_dir.mkdir(parents=True, exist_ok=True)
                dest = version_dir / f"{task.name}{source.suffix}"
                shutil.copy2(source, dest)
                srs_count += 1

        # 收集需求池和语义单元
        for pool_file in task_output_dir.glob("pool_iter_*.txt"):
            version_dir = units_collection_dir / pool_file.stem
            version_dir.mkdir(parents=True, exist_ok=True)
            dest = version_dir / f"{task.name}{pool_file.suffix}"
            shutil.copy2(pool_file, dest)
            units_count += 1
        for special in ["no-explore-clarify.txt", "no-clarify.txt"]:
            source = task_output_dir / special
            if source.exists():
                version_dir = units_collection_dir / source.stem
                version_dir.mkdir(parents=True, exist_ok=True)
                dest = version_dir / f"{task.name}{source.suffix}"
                shutil.copy2(source, dest)
                units_count += 1

    return srs_count, units_count


def write_time_statistics_csv(
    output_base_dir: Path,
    task_names: Sequence[str],
    results: Sequence[TaskResult],
) -> Optional[Path]:
    """写入耗时统计 CSV，保留历史数据"""

    csv_path = output_base_dir / "耗时统计-总耗时.csv"
    existing_task_order: List[str] = []
    existing_stage_order: List[str] = []
    combined_data: Dict[str, Dict[str, float]] = {}

    # 读取历史数据
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = list(csv.reader(csv_file))
        if reader:
            header = reader[0]
            existing_task_order = header[1:]
            for row in reader[1:]:
                if not row:
                    continue
                stage_name = row[0].strip()
                if not stage_name:
                    continue
                existing_stage_order.append(stage_name)
                stage_map = combined_data.setdefault(stage_name, {})
                for idx, task_name in enumerate(existing_task_order):
                    cell_idx = idx + 1
                    if cell_idx >= len(row):
                        continue
                    value = row[cell_idx].strip()
                    if not value:
                        continue
                    try:
                        stage_map[task_name] = float(value)
                    except ValueError:
                        continue

    # 合并本次运行的阶段耗时
    for result in results:
        if not result.stage_timings:
            continue
        for stage_name, duration in result.stage_timings.items():
            stage_map = combined_data.setdefault(stage_name, {})
            stage_map[result.name] = duration

    if not combined_data:
        print("未收集到阶段耗时信息，跳过写入耗时统计文件")
        return None

    final_task_order = list(dict.fromkeys(existing_task_order + list(task_names)))

    # 生成阶段顺序：优先使用历史顺序，再补充新阶段
    remaining_stages = set(combined_data.keys())
    stage_order: List[str] = []
    for stage_name in existing_stage_order:
        if stage_name in remaining_stages:
            stage_order.append(stage_name)
            remaining_stages.remove(stage_name)

    preferred = ["no-explore-clarify", "no-clarify"]
    for stage_name in preferred:
        if stage_name in remaining_stages:
            stage_order.append(stage_name)
            remaining_stages.remove(stage_name)

    iter_stages = sorted(
        [
            stage_name
            for stage_name in list(remaining_stages)
            if stage_name.startswith("iter") and stage_name[4:].isdigit()
        ],
        key=lambda name: int(name.replace("iter", "")),
    )
    for stage_name in iter_stages:
        stage_order.append(stage_name)
        remaining_stages.remove(stage_name)

    for stage_name in sorted(remaining_stages):
        stage_order.append(stage_name)

    if not stage_order:
        print("阶段名称列表为空，跳过写入耗时统计文件")
        return None

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["阶段名称", *final_task_order])
        for stage_name in stage_order:
            row = [stage_name]
            stage_map = combined_data.get(stage_name, {})
            for task_name in final_task_order:
                value = stage_map.get(task_name)
                row.append(f"{value:.2f}" if value is not None else "")
            writer.writerow(row)

    return csv_path


def write_summary(
    output_base_dir: Path,
    args: argparse.Namespace,
    results: Sequence[TaskResult],
    total_time: float,
    start_time: datetime,
    config,
) -> Path:
    """写入摘要文件"""
    summary_file = output_base_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    success_count = sum(1 for r in results if r.success and not r.skipped)
    skipped_count = sum(1 for r in results if r.skipped)
    fail_count = sum(1 for r in results if not r.success)
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("srs-gen2 批处理摘要\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"开始时间：{start_time:%Y-%m-%d %H:%M:%S}\n")
        f.write(f"总耗时：{total_time:.2f} 秒\n")
        f.write(f"d-orig 目录：{Path(args.d_orig_dir).resolve()}\n")
        f.write(f"r-base 目录：{Path(args.r_base_dir).resolve()}\n")
        f.write(f"d-base 目录：{Path(args.d_base_dir).resolve()}\n")
        f.write(f"输出目录：{output_base_dir.resolve()}\n")
        f.write(f"并行度：{args.parallel}\n")
        f.write(f"启动延迟：{args.startup_delay} 秒（增长系数：{args.startup_delay_growth}）\n")
        f.write(f"最大重试次数：{args.max_retries}，重试延迟：{args.retry_delay} 秒\n")
        f.write(f"跳过已生成：{'是' if args.skip_existing else '否'}\n")
        if args.rho is not None:
            f.write(f"rho（覆盖）：{args.rho}\n")
        else:
            f.write(f"rho（配置）：{config.rho}\n")
        if args.max_outer_iter is not None:
            f.write(f"max_outer_iter（覆盖）：{args.max_outer_iter}\n")
        else:
            f.write(f"max_outer_iter（配置）：{config.max_outer_iter}\n")
        if args.max_inner_iter is not None:
            f.write(f"max_inner_iter（覆盖）：{args.max_inner_iter}\n")
        else:
            f.write(f"max_inner_iter（配置）：{config.max_inner_iter}\n")
        f.write(f"文件扩展名：{', '.join(args.file_extensions)}\n")
        if args.files:
            f.write(f"指定文件：{', '.join(args.files)}\n")
        f.write("\n执行统计：\n")
        f.write(f"  总任务数：{len(results)}\n")
        f.write(f"  成功：{success_count}\n")
        if skipped_count:
            f.write(f"  跳过：{skipped_count}\n")
        f.write(f"  失败：{fail_count}\n")
        avg_time = total_time / success_count if success_count else 0.0
        if avg_time:
            f.write(f"  平均耗时：{avg_time:.2f} 秒/任务（成功任务）\n")
        f.write("\n任务明细：\n")
        for result in results:
            status = "成功" if result.success else "失败"
            if result.skipped:
                status = "跳过"
            f.write(f"- {result.name}: {status}，{result.message}\n")
    return summary_file


def main() -> None:
    """命令入口"""
    parser = argparse.ArgumentParser(description="srs-gen2 批处理脚本")
    parser.add_argument("--d-orig-dir", required=True, help="原始 SRS 文档目录")
    parser.add_argument("--r-base-dir", required=True, help="需求基础文档目录")
    parser.add_argument("--d-base-dir", required=True, help="基线 SRS 文档目录")
    parser.add_argument("--output-dir", default="output", help="输出目录（默认：output）")
    parser.add_argument("--parallel", type=int, default=1, help="并行度（默认：1，串行）")
    parser.add_argument("--rho", type=float, help="覆盖配置文件中的 rho")
    parser.add_argument("--max-outer-iter", type=int, help="覆盖配置文件中的 max_outer_iter")
    parser.add_argument("--max-inner-iter", type=int, help="覆盖配置文件中的 max_inner_iter")
    parser.add_argument(
        "--file-extensions",
        nargs="+",
        default=[".txt", ".md", ".markdown"],
        help="要处理的文件扩展名（默认：.txt .md .markdown）",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="仅处理指定文件（可包含或不包含扩展名）",
    )
    parser.add_argument("--max-retries", type=int, default=5, help="最大重试次数（默认：5）")
    parser.add_argument("--retry-delay", type=float, default=10.0, help="重试前等待时间（秒）")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已成功的任务")
    parser.add_argument("--startup-delay", type=float, default=0.0, help="任务提交延迟（秒）")
    parser.add_argument("--startup-delay-growth", type=float, default=0.0, help="启动延迟增长系数")
    args = parser.parse_args()

    if args.parallel < 1:
        print("错误：--parallel 必须大于等于 1", file=sys.stderr)
        sys.exit(1)

    config = get_config()

    d_orig_dir = Path(args.d_orig_dir)
    r_base_dir = Path(args.r_base_dir)
    d_base_dir = Path(args.d_base_dir)
    for directory, label in [
        (d_orig_dir, "d-orig"),
        (r_base_dir, "r-base"),
        (d_base_dir, "d-base"),
    ]:
        if not directory.exists() or not directory.is_dir():
            print(f"错误：{label} 目录不存在或不是目录：{directory}", file=sys.stderr)
            sys.exit(1)

    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    tasks = find_matching_files(
        d_orig_dir=d_orig_dir,
        r_base_dir=r_base_dir,
        d_base_dir=d_base_dir,
        extensions=args.file_extensions,
        specified_files=args.files,
    )

    if not tasks:
        print("错误：未找到任何输入文件", file=sys.stderr)
        sys.exit(1)

    r_base_dest = output_base_dir / "srs_collection" / "r_base"
    copied_r_base = copy_task_documents(tasks, "r_base", r_base_dest)
    print(f"已复制 {copied_r_base} 个 r-base 文档到：{r_base_dest}")

    d_base_dest = output_base_dir / "srs_collection" / "d_base"
    copied_d_base = copy_task_documents(tasks, "d_base", d_base_dest)
    print(f"已复制 {copied_d_base} 个 d-base 文档到：{d_base_dest}")

    print(f"找到 {len(tasks)} 个任务，将输出到 {output_base_dir.resolve()}")
    print(f"并行度：{args.parallel}，跳过已存在：{'是' if args.skip_existing else '否'}")

    task_map: Dict[str, TaskInput] = {task.name: task for task in tasks}
    results: List[TaskResult] = []
    start_time = datetime.now()
    total_timer = time.time()

    expected_iterations = args.max_outer_iter

    if args.parallel == 1:
        for idx, task in enumerate(tasks):
            apply_startup_delay(args.startup_delay, args.startup_delay_growth, idx)
            print(f"[{datetime.now():%H:%M:%S}] 开始处理：{task.name}")
            result = run_single_task(
                task=task,
                output_base_dir=output_base_dir,
                rho=args.rho,
                max_outer_iter=args.max_outer_iter,
                max_inner_iter=args.max_inner_iter,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
                skip_existing=args.skip_existing,
                expected_iterations=expected_iterations,
            )
            results.append(result)
            status = "✓" if result.success else "✗"
            print(f"[{datetime.now():%H:%M:%S}] {status} {result.name}：{result.message}")
    else:
        print("并行模式执行中...")
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            future_to_name = {}
            for idx, task in enumerate(tasks):
                apply_startup_delay(args.startup_delay, args.startup_delay_growth, idx)
                future = executor.submit(
                    run_single_task,
                    task,
                    output_base_dir,
                    args.rho,
                    args.max_outer_iter,
                    args.max_inner_iter,
                    args.max_retries,
                    args.retry_delay,
                    args.skip_existing,
                    expected_iterations,
                )
                future_to_name[future] = task.name
            completed = 0
            for future in as_completed(future_to_name):
                completed += 1
                result = future.result()
                results.append(result)
                status = "✓" if result.success else "✗"
                print(f"[{datetime.now():%H:%M:%S}] [{completed}/{len(tasks)}] {status} {result.name}：{result.message}")

    total_time = time.time() - total_timer
    success_count = sum(1 for r in results if r.success and not r.skipped)
    skipped_count = sum(1 for r in results if r.skipped)
    fail_count = sum(1 for r in results if not r.success)

    print()
    print("=" * 60)
    print("执行统计")
    print("=" * 60)
    print(f"总任务数：{len(results)}")
    print(f"成功：{success_count}")
    if skipped_count:
        print(f"跳过：{skipped_count}")
    print(f"失败：{fail_count}")
    print(f"总耗时：{total_time:.2f} 秒")
    if success_count:
        print(f"平均耗时：{(total_time / success_count):.2f} 秒/任务（仅计算成功任务）")

    summary_path = write_summary(
        output_base_dir=output_base_dir,
        args=args,
        results=results,
        total_time=total_time,
        start_time=start_time,
        config=config,
    )
    print(f"摘要文件写入：{summary_path}")

    srs_count, units_count = collect_outputs(
        output_base_dir=output_base_dir,
        task_map=task_map,
        results=results,
    )
    if srs_count:
        print(f"已收集 {srs_count} 个 SRS 文档")
    if units_count:
        print(f"已收集 {units_count} 份语义单元/需求池文件")

    time_stats_path = write_time_statistics_csv(
        output_base_dir=output_base_dir,
        task_names=[task.name for task in tasks],
        results=results,
    )
    if time_stats_path:
        print(f"耗时统计文件写入：{time_stats_path}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
