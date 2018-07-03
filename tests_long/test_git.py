import sys
sys.path.append('..')

from cannon.git_controller import GitController


if __name__ == '__main__':
    git = GitController('./test_repo', './experiment_repo')
    h = git.commit_experiment()
    print("commit hash: {}".format(h))