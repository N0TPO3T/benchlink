"""Benchmark runners — auto-register on import."""

from benchlink.benchmarks.manifeel_runner import ManiFeelRunner
from benchlink.benchmarks.manifeel_sim_runner import ManiFeelSimRunner
from benchlink.benchmarks.libero_runner import LiberoRunner
from benchlink.benchmarks.maniskill_runner import ManiSkillRunner
from benchlink.benchmarks.robotwin_runner import RoboTwinRunner
from benchlink.benchmarks.droid_sim_runner import DroidSimRunner
from benchlink.benchmarks.anytouch_probe_runner import AnyTouchProbeRunner
from benchlink.benchmarks.unitac_ecf_runner import UniTacECFRunner

from benchlink.registry import register_benchmark

register_benchmark("manifeel", ManiFeelRunner)
register_benchmark("manifeel_sim", ManiFeelSimRunner)
register_benchmark("libero", LiberoRunner)
register_benchmark("maniskill", ManiSkillRunner)
register_benchmark("robotwin", RoboTwinRunner)
register_benchmark("droid_sim", DroidSimRunner)
register_benchmark("anytouch_probe", AnyTouchProbeRunner)
register_benchmark("unitac_ecf", UniTacECFRunner)
