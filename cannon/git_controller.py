import os
import subprocess


class GitController():
    """
        Logging utilities to save the source code used to perform an experiment.
        Exploits git to save the source into a separate repository used only
        for logging puposes.

        Args:
            - git_root: original git repository
            - exp_log_root: experiment repository
    """

    def __init__(self, git_root, exp_log_root):
        self.git_root = git_root
        self.exp_log_root = exp_log_root

    def commit_experiment(self, msg=None):
        """
        Commit an experiment into the experiment repository.
        Args:
            - msg: commit message (optional)
        Returns:
            commit hash
        """
        cwd = os.getcwd()

        subprocess.call("cp -r {}/src {}/src".format(self.git_root, self.exp_log_root), shell=True)
        os.chdir(self.exp_log_root)
        subprocess.call("git add .", shell=True)
        subprocess.call("git commit -m {}".format('\"automatic experiment commit\"'), shell=True)

        commit_hash = subprocess.check_output("git rev-parse HEAD", shell=True)
        commit_hash = commit_hash.decode("utf8").strip()
        os.chdir(cwd)
        return commit_hash
