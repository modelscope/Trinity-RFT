# from beast_logger import print_dict
import subprocess
import argparse
import shutil
import time
import sys
import os
from dotenv import load_dotenv

load_dotenv()

BACK_TARGETS = os.environ.get(
    'BACK_TARGETS',
    'trinity/explorer'
).split(',')


def parse_args():
    parser = argparse.ArgumentParser(description='BA Launcher')
    parser.add_argument(
        '--target',
        type=str,
        default='trinity',
        required=False,
        help='Target script to run (default: trinity)'
    )
    parser.add_argument('--conf',
        type=str,
        default="",
        required=False,
        help='Path to configuration file'
    )
    parser.add_argument('--db',
        type=str,
        default="",
        required=False,
        help='Path to configuration file'
    )
    parser.add_argument('--with-ray',
        action='store_true',  # Changed from store_true to action='store_true'
        default=False,
        help='Launch ray'
    )
    parser.add_argument('--with-appworld',
        action='store_true',  # Changed from store_true to action='store_true'
        default=False,
        help='Launch appworld'
    )
    parser.add_argument('--with-appworld2',
        action='store_true',  # Changed from store_true to action='store_true'
        default=False,
        help='Launch appworld2'
    )
    parser.add_argument('--with-webshop',
        action='store_true',  # Changed from store_true to action='store_true'
        default=False,
        help='Launch webshop'
    )
    parser.add_argument('--with-logview',
        action='store_true',  # Changed from store_true to action='store_true'
        default=False,
        help='Launch logview'
    )
    parser.add_argument('--with-crafters',
        action='store_true',  # Changed from store_true to action='store_true'
        default=False,
        help='Launch Crafters Env Simulation'
    )
    parser.add_argument('--reboot',
        action='store_true',  # Changed from store_true to action='store_true'
        default=False,
        help='reboot flag'
    )

    return parser.parse_args()

def check_debugpy_version():
    """
    检查 debugpy 模块版本是否 >= 1.8.0
    如果未安装或版本过低，抛出 RuntimeError
    """
    try:
        import debugpy
    except ImportError:
        raise RuntimeError(
            "Module 'debugpy>=1.8.0' cannot be loaded. "
            "Ray Debugpy Debugger will not work without 'debugpy>=1.8.0' installed. "
            "Install this module using 'pip install debugpy>=1.8.0'"
        )

    # 检查版本
    version = getattr(debugpy, '__version__', '0.0.0')
    from packaging import version as packaging_version

    if packaging_version.parse(version) < packaging_version.parse('1.8.0'):
        raise RuntimeError(
            f"debugpy version {version} is too old. "
            "Ray Debugpy Debugger requires 'debugpy>=1.8.0'. "
            "Upgrade using 'pip install debugpy>=1.8.0'"
        )

    print(f"✓ debugpy version {version} meets requirement (>=1.8.0)")

check_debugpy_version()
def main():
    args = parse_args()

    if args.conf:
        yaml_path = args.conf
        assert yaml_path.endswith('.yaml'), "Configuration file must be a YAML file"
        exp_base = os.path.dirname(args.conf)

        if os.path.exists(exp_base):

            ## 0. read yaml (get trainer.experiment_name)
            import yaml
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
            exp_name = config.get('trainer').get('experiment_name')
            if exp_name is None or exp_name == 'read_yaml_name':
                if exp_name is not None: exp_name = exp_name.replace('|', '-')
                exp_name = os.path.basename(yaml_path).replace('.yaml', '')
            else:
                exp_name = exp_name.replace('|', '-')

            print('----------------------------------------')
            backup_dir = os.path.join('launcher_record', exp_name, 'backup')
            yaml_backup_dst = os.path.join('launcher_record', exp_name, 'yaml_backup.yaml')
            exe_yaml_path = yaml_backup_dst
            exe_exp_base = os.path.dirname(yaml_backup_dst)
            print('Experiment Name:', exp_name)
            print('Experiment Backup Dir:', backup_dir)
            print('Experiment Yaml Dir:', yaml_backup_dst)
            print('----------------------------------------')
            time.sleep(2)

            ## 1. check exp_base/backup exist
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            else:
                total_seconds = 10
                for i in range(total_seconds):
                    print(f"\rWarning: backup directory already exists, we will automatically ignore this after {total_seconds - i} seconds...", end="", flush=True)
                    time.sleep(1)

            ## 2. copy files to backup
            for backup_target in BACK_TARGETS:
                print(f"Copying {backup_target} to {os.path.join(backup_dir, os.path.basename(backup_target))}")
                shutil.copytree(backup_target, os.path.join(backup_dir, os.path.basename(backup_target)), dirs_exist_ok=True)

            ## 3. copy yaml to backup
            yaml_backup_src = yaml_path
            shutil.copyfile(yaml_backup_src, yaml_backup_dst)

            ## 4. edit new yaml
            yaml_path = yaml_backup_dst
            # now, replace the trainer.experiment_name
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
            # config['trainer']['experiment_name'] = exp_name
            with open(yaml_path, 'w') as file:
                yaml.dump(config, file)

        else:
            raise FileNotFoundError(f"Configuration file not found: {exp_base}")

        env = os.environ.copy()
        if args.db:
            env["RAY_DEBUG_POST_MORTEM"] = "1"
            env["DEBUG_TAGS"] = args.db
            env["RAY_record_task_actor_creation_sites"] =  "true"
            print("Debug mode is ON")
        else:
            print("Debug mode is OFF")
    else:
        assert args.with_appworld or args.with_webshop or args.with_logview or args.with_crafters, "You must at least do something."
    if args.with_ray:
        from agentopia.utils.smart_daemon import LaunchCommandWhenAbsent
        ray_env = {}
        if args.db:
            ray_env["RAY_DEBUG_POST_MORTEM"] = "1"
            ray_env["DEBUG_TAGS"] = args.db
            ray_env["RAY_record_task_actor_creation_sites"] =  "true"
        companion = LaunchCommandWhenAbsent(
            full_argument_list=[
                f"source ./.venv/bin/activate && ray start --head && sleep infinity"
            ],
            dir='./',
            tag="ray_service",
            use_pty=True
        )
        companion.launch(
            launch_wait_time=1800,
            success_std_string="Ray runtime started",
            env_dict=ray_env,
        )
    if args.with_appworld:
        from agentopia.utils.smart_daemon import LaunchCommandWhenAbsent
        companion = LaunchCommandWhenAbsent(
            full_argument_list=[
                f"source /mnt/data/taoshuchang.tsc/anaconda3/etc/profile.d/conda.sh && conda activate appworld && bash -i EnvService/env_sandbox/appworld.sh"
            ],
            dir='/mnt/data/taoshuchang.tsc/beyondagent',
            tag="appworld_env_service",
            use_pty=True
        )
        companion.launch(
            launch_wait_time=1800,
            success_std_string="Starting server on",
        )
    if args.with_crafters:
        from agentopia.utils.smart_daemon import LaunchCommandWhenAbsent
        crafters_path = os.environ.get('CRAFTERS_PATH')
        crafters_script = os.environ.get('CRAFTERS_SCRIPT')
        crafters_conda = os.environ.get('CRAFTERS_CONDA')
        if crafters_path and os.path.exists(crafters_path):
            companion = LaunchCommandWhenAbsent(
                full_argument_list=[
                    f"source {crafters_conda} && conda activate balrog && bash -i {crafters_script}"
                ],
                dir=crafters_path,
                tag="crafters_env_service",
                use_pty=True
            )
            companion.launch(
                launch_wait_time=1800,
                success_std_string="Starting server on",
            )
        else:
            raise RuntimeError("EnvService not found")
    if args.with_webshop:
        from agentopia.utils.smart_daemon import LaunchCommandWhenAbsent
        webshop_path = os.environ.get('WEBSHOP_PATH')
        webshop_python = os.environ.get('WEBSHOP_PYTHON')
        webshop_port = os.environ.get('WEBSHOP_PORT', '1907')
        webshop_env_port = os.environ.get('WEBSHOP_ENV_PORT', '8080')
        java_home = os.environ.get('JAVA_HOME')
        java_ld_library_path = os.environ.get('JAVA_LD_LIBRARY_PATH')
        search_engine_path = os.environ.get('SEARCH_ENGINE_PATH')
        webshop_root = os.environ.get('WEBSHOP_ROOT')
        items_attr_path = os.environ.get('ITEMS_ATTR_PATH')
        items_file_path = os.environ.get('ITEMS_FILE_PATH')
        pythonpath = os.environ.get('PYTHONPATH')
        if webshop_path and os.path.exists(webshop_path):
            companion = LaunchCommandWhenAbsent(
                full_argument_list=[
                    webshop_python,
                    '-m',
                    'env_sandbox.environments.webshop.SimServer_launch',
                    "--portal",
                    "127.0.0.1",
                    "--port",
                    webshop_port,
                ],
                dir=webshop_path,
                tag="webshop_sim_server"
            )

            companion.launch(launch_wait_time=1800, success_std_string="Uvicorn running on", env_dict={
                "JAVA_HOME": java_home,
                "JAVA_LD_LIBRARY_PATH": java_ld_library_path,
                "search_engine_path": search_engine_path,
                "webshop_root": webshop_root,
                "ITEMS_ATTR_PATH": items_attr_path,
                "ITEMS_FILE_PATH": items_file_path,
                "PYTHONPATH": pythonpath
            }, force_restart=args.reboot)

            companion = LaunchCommandWhenAbsent(
                full_argument_list=[
                    webshop_python,
                    '-m',
                    'env_sandbox.env_service',
                    "--env",
                    "webshop",
                    "--portal",
                    "127.0.0.1",
                    "--port",
                    webshop_env_port,
                ],
                dir=webshop_path,
                tag="webshop_env_service"
            )
            companion.launch(launch_wait_time=1800,success_std_string="Uvicorn running on", env_dict={
                "JAVA_HOME": java_home,
                "JAVA_LD_LIBRARY_PATH": java_ld_library_path
            }, force_restart=args.reboot)
        else:
            raise RuntimeError("EnvService not found")




    if args.with_logview:
        from agentopia.utils.smart_daemon import LaunchCommandWhenAbsent
        logview_nvm_dir = os.environ.get('LOGVIEW_NVM_DIR')
        logview_nvm_bin = os.environ.get('LOGVIEW_NVM_BIN')
        logview_path = os.environ.get('LOGVIEW_PATH')
        companion = LaunchCommandWhenAbsent(
            full_argument_list=[
                sys.executable,
                '-m',
                'web_display.go',
            ],
            dir='./',
            tag="logview"
        )
        companion.launch(launch_wait_time=1800,success_std_string="Server running on", env_dict={
            'NVM_DIR': logview_nvm_dir,
            'NVM_BIN': logview_nvm_bin,
            'PATH': logview_path + os.environ.get('PATH', '')
        })

    if args.conf:
        # let's begin the training process
        cmd = [
            sys.executable,
            '-m',
            'trinity.cli.launcher',
            # python -m trinity.cli.launcher run --config xxx.yaml
            'run',
            '--config', yaml_backup_dst
        ]

        if args.with_logview:
            env.update({
                'BEST_LOGGER_WEB_SERVICE_URL': os.environ.get('BEST_LOGGER_WEB_SERVICE_URL', 'http://127.0.0.1:8181/')
            })

        try:
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=os.path.abspath('./'), env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error running subprocess: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()