"""
Unit test for hal logging
"""

import logging
import logging.config

from halucinator import hal_log


class TestHalLog:
    def test_hal_logger_set_correctly(self):
        assert hal_log.getHalLogger() == logging.getLogger(hal_log.HAL_LOGGER)


CFG_WITHOUT_HALUCINATOR_PARENT = """\
[loggers]
keys=root,halucinator.main,HAL_LOG

[handlers]
keys=consoleHandler

[formatters]
keys=sampleFormatter

[logger_root]
level=ERROR
handlers=consoleHandler

[logger_halucinator.main]
level=INFO
handlers=consoleHandler
propagate=0
qualname=halucinator.main

[logger_HAL_LOG]
level=INFO
handlers=consoleHandler
propagate=0
qualname=HAL_LOG

[handler_consoleHandler]
class=StreamHandler
level=DEBUG
formatter=sampleFormatter
args=(sys.stdout,)

[formatter_sampleFormatter]
format=%(name)s|%(levelname)s|  %(message)s
"""


class TestCwdLogConfigDoesNotSilenceHalucinator:
    """A `logging.cfg` in the working directory must not disable the
    `halucinator.*` loggers.

    Every module in the package does `log = logging.getLogger(__name__)` at
    IMPORT time, so by the time setLogConfig() runs they are all "existing"
    loggers. The CWD branch passed disable_existing_loggers=True while the
    packaged-default branch passed False, so merely having a logging.cfg in
    the working directory switched off every `halucinator.*` logger that is
    not a child of a logger the cfg names.

    `tutorial/logging.cfg` -- the file most likely to be picked up, because it
    sits in a directory people run from -- is exactly this shape: it declares
    a [logger_halucinator] section but never lists `halucinator` in its
    [loggers] keys, so the parent is never created and every backend logger
    is an unconfigured existing logger. The run still worked; you just went
    blind, which is what makes it expensive.
    """

    LOGGER = "halucinator.backends.unicorn_backend"

    def teardown_method(self):
        # fileConfig() mutates process-global logging state; put the packaged
        # config back so later tests are unaffected.
        logging.config.fileConfig(fname=hal_log.DEFAULT_LOG_CONFIG,
                                  disable_existing_loggers=False)
        logging.getLogger(self.LOGGER).disabled = False

    def _write_cfg(self, tmp_path):
        p = tmp_path / hal_log.LOG_CONFIG_NAME
        p.write_text(CFG_WITHOUT_HALUCINATOR_PARENT)
        return p

    def test_cwd_config_leaves_backend_logger_enabled(self, tmp_path,
                                                      monkeypatch):
        self._write_cfg(tmp_path)
        monkeypatch.chdir(tmp_path)
        lg = logging.getLogger(self.LOGGER)
        lg.disabled = False

        hal_log.setLogConfig()

        assert lg.disabled is False, (
            "a logging.cfg in the CWD disabled halucinator's own loggers -- "
            "every backend diagnostic is silently lost")

    def test_backend_warning_still_reaches_a_handler(self, tmp_path,
                                                     monkeypatch, capsys):
        """The end-to-end symptom: a backend warning (the shape _x86_in_hook
        emits before falling through to `val = 0`) must still be emitted."""
        self._write_cfg(tmp_path)
        monkeypatch.chdir(tmp_path)
        lg = logging.getLogger(self.LOGGER)
        lg.disabled = False

        hal_log.setLogConfig()
        capsys.readouterr()
        lg.warning("x86 IN  port=0x%x: handler raised %s", 0x3F8, "boom")

        assert "handler raised boom" in capsys.readouterr().out
