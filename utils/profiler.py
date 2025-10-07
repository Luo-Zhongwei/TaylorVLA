import numpy as np
from collections import defaultdict

class ModelProfiler:
    """
    一个支持双模式（详细/简要）和多重实验分析的终极性能分析工具。
    - 'detailed' 模式: 分析到 run -> trial -> process -> step -> layer 五个层级。
    - 'simple' 模式: 只分析 run -> trial -> process 层级的核心事件 (如 Vision, Language, Action)。
    """
    def __init__(self, mode: str = 'detailed'):
        if mode not in ['detailed', 'simple']:
            raise ValueError("Mode 必须是 'detailed' 或 'simple'")
        self.profiling_mode = mode

        # 为两种模式分别准备数据结构
        self.detailed_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
        self.simple_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

        # 当前上下文状态
        self.current_run = -1
        self.current_trial = -1
        self.current_process = -1
        self.current_step = -1

    # --- 上下文管理方法 ---
    def start_experiment_run(self, run_idx: int):
        self.current_run = run_idx
        self.current_trial = -1; self.current_process = -1; self.current_step = -1

    def start_trial(self, trial_idx: int):
        if self.current_run == -1: return
        self.current_trial = trial_idx
        self.current_process = -1; self.current_step = -1

    def start_diffusion_process(self, process_idx: int):
        if self.current_trial == -1: return
        self.current_process = process_idx
        self.current_step = -1

    # --- 数据记录方法 ---
    def start_diffusion_step(self, step_idx: int):
        """仅在 'detailed' 模式下有效"""
        if self.profiling_mode != 'detailed' or self.current_process == -1: return
        self.current_step = step_idx

    def record(self, component: str, time_ms: float):
        """仅在 'detailed' 模式下有效，记录最底层的 layer/component 时间"""
        if self.profiling_mode != 'detailed' or self.current_step == -1: return
        self.detailed_data[self.current_run][self.current_trial][self.current_process][self.current_step][component].append(time_ms)

    def record_process_events(self, events_dict: dict):
        """
        仅在 'simple' 模式下有效。
        一次性记录一个 process 内发生的多个高层级事件。
        :param events_dict: 一个包含事件名称和时间的字典, e.g., {'Vision': 10.2, 'Action': 50.5}
        """
        if self.profiling_mode != 'simple' or self.current_process == -1: return
        for event_name, time_ms in events_dict.items():
            if time_ms is not None:
                self.simple_data[self.current_run][self.current_trial][self.current_process][event_name].append(time_ms)

    # --- 公共报告接口 ---
    def generate_report(self):
        """根据当前模式生成对应的报告"""
        if self.profiling_mode == 'simple':
            return self._generate_simple_report()
        else:
            return self._generate_detailed_report()

    def display_report(self, report: dict):
        """根据报告类型显示对应的格式"""
        if not report or "error" in report:
            error_msg = report.get("error", "没有收集到任何数据。")
            print(f"无法生成报告: {error_msg}")
            return
        if report.get("report_type") == 'simple':
            self._display_simple_report(report)
        else:
            self._display_detailed_report(report)
            
    # --- Simple 模式的报告实现 ---
    def _generate_simple_report(self):
        if not self.simple_data: return {"error": "简要模式下没有收集到任何数据。"}
        run_reports, overall_aggregator = {}, defaultdict(list)
        for run_idx, trials in self.simple_data.items():
            run_events = defaultdict(list)
            for _, processes in trials.items():
                for _, events in processes.items():
                    for event_name, times in events.items():
                        run_events[event_name].extend(times)
            run_summary = {}
            for event_name, all_times in run_events.items():
                if not all_times: continue
                mean_time = np.mean(all_times)
                run_summary[event_name] = {"mean": mean_time, "std": np.std(all_times), "count": len(all_times)}
                overall_aggregator[event_name].append(mean_time)
            run_reports[run_idx] = run_summary
        overall_summary = {}
        for event_name, run_means in overall_aggregator.items():
            if not run_means: continue
            overall_summary[event_name] = {"mean_of_means": np.mean(run_means), "std_of_means": np.std(run_means), "count": len(run_means)}
        return {"report_type": "simple", "overall_summary": overall_summary, "per_experiment_run_summary": run_reports}

    def _display_simple_report(self, report: dict):
        print("\n" + "="*60); print("📊 简要性能分析报告 (V-L-A 模式)"); print("="*60)
        summary = report.get("overall_summary")
        if summary:
            print("\n--- 1. 总体平均性能 (所有 Experiment Run 的平均值) ---")
            for event, stats in sorted(summary.items()): print(f"  - {event:<20}: {stats['mean_of_means']:.3f} ms (标准差: {stats['std_of_means']:.3f} ms)")
        run_summary = report.get("per_experiment_run_summary", {}).get(0)
        if run_summary:
            print("\n--- 2. 单次 Experiment Run 平均性能 (示例: Run 0) ---")
            for event, stats in sorted(run_summary.items()): print(f"  - {event:<20}: {stats['mean']:.3f} ms (共 {stats['count']} 次记录)")
        print("="*60 + "\n")

    # --- Detailed 模式的报告实现 (完整版) ---
    def _generate_detailed_report(self):
        if not self.detailed_data: return {"error": "详细模式下没有收集到任何数据。"}
        per_layer_data, overall_stats_aggregator = defaultdict(lambda: defaultdict(list)), defaultdict(list)
        run_reports, trial_reports, process_reports = {}, {}, {}
        for run_idx in sorted(self.detailed_data.keys()):
            run_total_times = defaultdict(list)
            for trial_idx in self.detailed_data[run_idx]:
                trial_total_times = defaultdict(list)
                for process_idx in self.detailed_data[run_idx][trial_idx]:
                    process_total_times = defaultdict(list)
                    for step_idx in self.detailed_data[run_idx][trial_idx][process_idx]:
                        for component, layer_times in self.detailed_data[run_idx][trial_idx][process_idx][step_idx].items():
                            process_total_times[component].extend(layer_times)
                            for layer_idx, time in enumerate(layer_times):
                                per_layer_data[component][layer_idx].append(time)
                    process_summary = {comp: {"mean": np.mean(t), "std": np.std(t), "count": len(t)} for comp, t in process_total_times.items() if t}
                    process_reports[(run_idx, trial_idx, process_idx)] = process_summary
                    for component, all_times in process_total_times.items(): trial_total_times[component].extend(all_times)
                trial_summary = {comp: {"mean": np.mean(t), "std": np.std(t), "count": len(t)} for comp, t in trial_total_times.items() if t}
                trial_reports[(run_idx, trial_idx)] = trial_summary
                for component, all_times in trial_total_times.items(): run_total_times[component].extend(all_times)
            run_summary = {}
            for component, all_times in run_total_times.items():
                if not all_times: continue
                mean_time = np.mean(all_times)
                run_summary[component] = {"mean": mean_time, "std": np.std(all_times), "count": len(all_times)}
                overall_stats_aggregator[component].append(mean_time)
            run_reports[run_idx] = run_summary
        per_layer_summary = defaultdict(dict)
        for component, layer_data in per_layer_data.items():
            for layer_idx, times in layer_data.items():
                if not times: continue
                per_layer_summary[component][layer_idx] = {"mean": np.mean(times), "std": np.std(times), "count": len(times)}
        overall_summary = {comp: {"mean_of_means": np.mean(m), "std_of_means": np.std(m), "count": len(m)} for comp, m in overall_stats_aggregator.items() if m}
        return {"report_type": "detailed", "overall_summary": overall_summary, "per_layer_summary": per_layer_summary,
                "per_experiment_run_summary": run_reports, "per_trial_summary": trial_reports, "per_diffusion_process_summary": process_reports}

    def _display_detailed_report(self, report: dict):
        print("\n" + "="*60); print("📊 高级性能分析报告 (多重实验详细版)"); print("="*60)
        summary = report.get("overall_summary")
        if summary:
            print("\n--- 1. 总体平均性能 (所有 Experiment Run 的平均值) ---")
            for comp, stats in sorted(summary.items()): print(f"  - {comp:<20}: {stats['mean_of_means']:.3f} ms (标准差: {stats['std_of_means']:.3f} ms)")
        per_layer_summary = report.get("per_layer_summary")
        if per_layer_summary:
            print("\n--- 2. 逐层平均性能 (所有过程、所有step的平均值) ---")
            for comp, layer_data in sorted(per_layer_summary.items()):
                layer_indices = sorted(layer_data.keys())
                if not layer_indices: continue
                print(f"  Component: [ {comp} ]")
                sample_indices = {layer_indices[0]}
                if len(layer_indices) > 2: sample_indices.add(layer_indices[len(layer_indices) // 2])
                if len(layer_indices) > 1: sample_indices.add(layer_indices[-1])
                for idx in sorted(list(sample_indices)): print(f"    - Layer {idx:<3}: {layer_data[idx]['mean']:.3f} ms (标准差: {layer_data[idx]['std']:.3f} ms)")
            if any(len(v) > 3 for v in per_layer_summary.values()): print("  (提示: 报告字典中包含所有层的详细数据)")
        run_summary = report.get("per_experiment_run_summary", {}).get(0)
        if run_summary:
            print("\n--- 3. 单次 Experiment Run 平均性能 (示例: Run 0) ---")
            for comp, stats in sorted(run_summary.items()): print(f"  - {comp:<20}: {stats['mean']:.3f} ms (共 {stats['count']} 次底层记录)")
        trial_summary = report.get("per_trial_summary", {}).get((0, 0))
        if trial_summary:
            print("\n--- 4. 单次 Trial 平均性能 (示例: Run 0, Trial 0) ---")
            for comp, stats in sorted(trial_summary.items()): print(f"  - {comp:<20}: {stats['mean']:.3f} ms (共 {stats['count']} 次底层记录)")
        process_summary = report.get("per_diffusion_process_summary", {}).get((0, 0, 0))
        if process_summary:
            print("\n--- 5. 单次 Diffusion 过程平均性能 (示例: Run 0, Trial 0, Process 0) ---")
            for comp, stats in sorted(process_summary.items()): print(f"  - {comp:<20}: {stats['mean']:.3f} ms (共 {stats['count']} 次底层记录)")
        print("="*60 + "\n")

# --- 全局实例 ---
# 在这里根据你的需要选择profiler的模式
# global_profiler = ModelProfiler(mode='detailed') 
global_profiler = ModelProfiler(mode='simple')





#this var use for debug ,for print class name
global_class_print_count=0
PRINT_LIMIT = 10 # 设置一个打印上限，比如10次