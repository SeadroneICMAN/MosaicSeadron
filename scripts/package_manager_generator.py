import os
import argparse
from typing import Sequence, List
from pathlib import Path

INSTALL_FILE : str = 'install.sh'
UNINSTALL_FILE : str = 'uninstall.sh'
SETUP_FILE : str = 'setup.py'

to_bool = lambda x : bool(int(x))

def generate_install_script(out_folder : str, packages : List[str], run : bool = False) -> None:
    install_template : str = 'python \'{}\' install\n'
    install_path : str = os.path.join(out_folder, INSTALL_FILE)

    with open(install_path, 'w') as dst:
        project_folder = os.path.dirname(out_folder)

        for package in packages:
            for path in Path(project_folder).rglob(os.path.join(package, SETUP_FILE)):
                dst.write( install_template.format(path) )

    if run:
        os.system(f'{install_path}')

def generate_uninstall_script(out_folder : str, packages : List[str], env_folder : str, run : bool = False) -> None:
    uninstall_template : str = 'rm -rf \'{}\'\n'
    build_folder : str = 'build'
    dist_folder : str = 'dist'
    setup_extension : str = '.egg-info'
    project_folder : str = os.path.dirname(out_folder)
    uninstall_path : str = os.path.join(out_folder, UNINSTALL_FILE)

    with open(uninstall_path, 'w') as dst:
        for path in Path(project_folder).glob(build_folder):
            dst.write(uninstall_template.format(path))
        for path in Path(project_folder).glob(dist_folder):
            dst.write(uninstall_template.format(path))
        for package in packages:
            for path in Path(project_folder).glob(f'{package}{setup_extension}'):
                dst.write(uninstall_template.format(path))
            for path in Path(os.path.join(project_folder, env_folder)).rglob(f'*{package}*'):
                dst.write(uninstall_template.format(path))

    if run:
        os.system(f'{uninstall_path}')

def main(argvs : Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env_folder", required = True)
    parser.add_argument("-p", "--packages", required = True)
    parser.add_argument("-d", "--destiny", required = False, default = os.path.dirname(__file__))
    parser.add_argument("-i", "--install", required = False, default = True, type = to_bool)
    parser.add_argument("-ri", "--run_install", required = False, default = False, type = to_bool)
    parser.add_argument("-u", "--uninstall", required = False, default = True, type = to_bool)
    parser.add_argument("-ru", "--run_uninstall", required = False, default = False, type = to_bool)

    args = parser.parse_args(argvs)

    if args.install:
        generate_install_script(args.destiny, args.packages.split(','), args.run_install)

    if args.uninstall:
        generate_uninstall_script(args.destiny, args.packages.split(','), args.env_folder, args.run_uninstall)

if __name__ == '__main__':
    main()