from pathlib import Path, PosixPath


def get_current_file_parent_path(file) -> PosixPath:
    return Path(file).parent.resolve()


if __name__ == '__main__':
    print(f'the current file parent path is: {get_current_file_parent_path(__file__)}')