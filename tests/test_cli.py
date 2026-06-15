"""Test CLI argument parsing."""

from benchlink.cli import build_parser


class TestCLIParsing:
    """Verify CLI argument parsing."""

    def test_list_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--list"])
        assert args.list is True

    def test_minimal_eval_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "--model", "fastwam",
            "--benchmark", "libero",
            "--checkpoint", "/path/to/ckpt",
            "--config", "configs/libero.yaml",
        ])
        assert args.model == "fastwam"
        assert args.benchmark == "libero"
        assert args.checkpoint == "/path/to/ckpt"
        assert args.config == "configs/libero.yaml"
        assert args.episodes == 5  # default

    def test_with_episodes(self):
        parser = build_parser()
        args = parser.parse_args([
            "--model", "rdp",
            "--benchmark", "manifeel",
            "--checkpoint", "/path/to/ckpt",
            "--config", "configs/default.yaml",
            "--episodes", "10",
        ])
        assert args.episodes == 10

    def test_with_overrides(self):
        parser = build_parser()
        args = parser.parse_args([
            "--model", "anytouch",
            "--benchmark", "anytouch_probe",
            "--checkpoint", "/path/to/ckpt",
            "--config", "configs/anytouch.yaml",
            "--data-root", "/custom/data",
            "--device", "cpu",
        ])
        assert args.data_root == "/custom/data"
        assert args.device == "cpu"
