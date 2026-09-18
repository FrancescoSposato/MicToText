"""Mermaid rendering through mermaid-cli (mmdc), natively or inside WSL2."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from mictotext.config import RenderConfig

_BASH_NOISE = ("cannot set terminal process group", "no job control in this shell")


def _decode(data: bytes) -> str:
    # wsl.exe's own messages are UTF-16LE; Linux tools emit UTF-8. Stripping NULs handles both.
    return data.decode("utf-8", errors="replace").replace("\x00", "")


def _clean_output(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines()
        if line.strip() and not any(noise in line for noise in _BASH_NOISE)
    ).strip()


def _render_result(process: subprocess.CompletedProcess, dst: Path) -> tuple[bool, str]:
    output = _clean_output(_decode(process.stderr) + "\n" + _decode(process.stdout))
    if process.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
        return True, output
    return False, output or f"mmdc exited with code {process.returncode}"


class NativeMmdcRenderer:
    """mmdc available on the PATH of the current OS (Windows, or Linux when run inside WSL)."""

    def __init__(self, executable: str, cfg: RenderConfig) -> None:
        self.executable = executable
        self.cfg = cfg
        self.name = f"mmdc (native: {executable})"

    @classmethod
    def detect(cls, cfg: RenderConfig) -> NativeMmdcRenderer | None:
        executable = shutil.which(cfg.mmdc_command)
        return cls(executable, cfg) if executable else None

    def render(self, src: Path, dst: Path) -> tuple[bool, str]:
        dst.unlink(missing_ok=True)
        command = [self.executable, "-i", str(src), "-o", str(dst), "-b", "white", "-s", str(self.cfg.scale)]
        if self.cfg.puppeteer_config:
            command += ["-p", self.cfg.puppeteer_config]
        try:
            process = subprocess.run(command, capture_output=True, stdin=subprocess.DEVNULL,
                                     timeout=self.cfg.timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, str(exc)
        return _render_result(process, dst)


class WslMmdcRenderer:
    """mmdc installed inside a WSL2 distro, invoked from Windows via wsl.exe."""

    def __init__(self, cfg: RenderConfig) -> None:
        self.cfg = cfg
        self._base = ["wsl.exe"] + (["-d", cfg.wsl_distro] if cfg.wsl_distro else [])
        self.name = f"mmdc via WSL ({cfg.wsl_distro or 'default distro'})"

    def _run(self, args: list[str], timeout: int) -> subprocess.CompletedProcess:
        return subprocess.run(self._base + ["--exec", *args], capture_output=True,
                              stdin=subprocess.DEVNULL, timeout=timeout)

    def _bash(self, command: str, timeout: int) -> subprocess.CompletedProcess:
        # Login + interactive shell so ~/.profile and ~/.bashrc (e.g. nvm PATH) are loaded.
        return self._run(["bash", "-lic", command], timeout)

    @classmethod
    def detect(cls, cfg: RenderConfig) -> WslMmdcRenderer | None:
        if sys.platform != "win32" or shutil.which("wsl.exe") is None:
            return None
        renderer = cls(cfg)
        try:
            process = renderer._bash(f"command -v {shlex.quote(cfg.mmdc_command)}", timeout=90)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return renderer if process.returncode == 0 else None

    def to_wsl_path(self, path: Path) -> str:
        path = path.resolve()
        try:
            process = self._run(["wslpath", "-a", str(path)], timeout=30)
            converted = _decode(process.stdout).strip()
            if process.returncode == 0 and converted.startswith("/"):
                return converted
        except (OSError, subprocess.TimeoutExpired):
            pass
        # Fallback for the standard automount layout: C:\x\y -> /mnt/c/x/y
        drive = path.drive.rstrip(":").lower()
        return f"/mnt/{drive}{path.as_posix()[len(path.drive):]}"

    def render(self, src: Path, dst: Path) -> tuple[bool, str]:
        dst.unlink(missing_ok=True)
        parts = [self.cfg.mmdc_command, "-i", self.to_wsl_path(src), "-o", self.to_wsl_path(dst),
                 "-b", "white", "-s", str(self.cfg.scale)]
        if self.cfg.puppeteer_config:
            parts += ["-p", self.cfg.puppeteer_config]
        command = " ".join(shlex.quote(part) for part in parts)
        try:
            process = self._bash(command, timeout=self.cfg.timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, str(exc)
        return _render_result(process, dst)


def build_renderer(cfg: RenderConfig) -> NativeMmdcRenderer | WslMmdcRenderer | None:
    """Pick a renderer according to cfg.renderer. None means HTML-only output."""
    if cfg.renderer == "html":
        return None
    if cfg.renderer in ("auto", "native"):
        native = NativeMmdcRenderer.detect(cfg)
        if native or cfg.renderer == "native":
            return native
    if cfg.renderer in ("auto", "wsl"):
        return WslMmdcRenderer.detect(cfg)
    return None
