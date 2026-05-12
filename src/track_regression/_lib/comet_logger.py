"""Largely inspired by / copied from https://github.com/samvanstroud/hepattn."""

import os

from lightning.pytorch.loggers import CometLogger


class MyCometLogger(CometLogger):
    """Wrap CometLogger to fix issues with CLI arguments.

    Overrides ``save_dir`` so that ``trainer.log_dir`` resolves to a
    per-experiment directory::

        <offline_directory>/<experiment_key>/

    This ensures checkpoints, configs, and other artefacts are cleanly
    separated by experiment rather than all landing in one shared folder.

    Also disables comet_ml's auto_param_logging / parse_args / log_graph
    hooks by default.  Those hooks walk every ``nn.Module`` for
    hyperparameter-looking attributes (e.g. ``sigma_init``, ``sigma_floor``,
    spline knot vectors, bin edges) and flood the Comet Hyperparameters
    tab with loss-submodule buffer names that are not actually run
    hyperparameters.  We log everything we want explicitly through
    ``save_hyperparameters`` and ``SaveConfig``.
    """

    def __init__(
        self,
        name: str,
        project_name: str = "ssm-track-regression",
        offline_directory: str | None = None,
        log_env_details: bool = True,
        auto_param_logging: bool = False,
        auto_metric_logging: bool = True,
        parse_args: bool = False,
        log_graph: bool = False,
        **kwargs,
    ):
        assert offline_directory is not None, "offline_directory must be specified for MyCometLogger"
        self._offline_directory = offline_directory
        super().__init__(
            name=name,
            project_name=project_name,
            offline_directory=offline_directory,
            log_env_details=log_env_details,
            auto_param_logging=auto_param_logging,
            auto_metric_logging=auto_metric_logging,
            parse_args=parse_args,
            log_graph=log_graph,
            **kwargs,
        )

    @property
    def save_dir(self) -> str:
        """Return a per-experiment directory using the Comet experiment key.

        Lightning's ``Trainer.log_dir`` uses ``logger.save_dir`` for
        non-TensorBoard loggers, so placing the experiment key here gives
        each run its own directory for checkpoints and metadata.
        """
        key = self.version  # experiment key (hex string)
        if key is not None:
            return os.path.join(self._offline_directory, key)
        return self._offline_directory
