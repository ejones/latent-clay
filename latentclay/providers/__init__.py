import re
import subprocess
import shlex


class ProcessRunner:
    def __init__(self, cmd, progress_re=None, env=None, device=None):
        self.cmd = cmd
        self.progress_re = progress_re and re.compile(progress_re)
        self.env = env
        self.device = device

    def run(self, params, report_progress):
        args = [
            arg.format(
                device=self.device,
                device_arg=f'--device {self.device}' if self.device else '',
                **params,
            )
            for arg in shlex.split(self.cmd)
        ]
        with subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=self.env
        ) as proc:
            for line in proc.stdout:
                print(repr(line))
                if self.progress_re:
                    match = self.progress_re.search(line)
                    print('->', match and repr(match.group()))
                    if match:
                        groups = match.groups()
                        report_progress(groups[0] if groups else match.group())
        if proc.returncode != 0:
            raise RuntimeError('Error in process:', self.cmd)

