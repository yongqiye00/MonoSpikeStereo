import os

from runtime import bootstrap_simulation_runtime

bootstrap_simulation_runtime()

from common.simulation_cli import (
    SimulationConfig,
    namespace_to_simulation_config,
    parse_legacy_crop,
    parse_simulation_args,
)
from pipelines.tartanair_lcd_simulation import run_simulation_LCD


Config = SimulationConfig
parse_crop = parse_legacy_crop


def entry(cfg: Config):
    if not cfg.lcd:
        raise ValueError("TartanAir simulation currently exposes the LCD branch.")

    crop_size = parse_crop(cfg.crop)
    scene = sorted(os.listdir(cfg.root))

    for s in scene:
        if not os.path.isdir(os.path.join(cfg.root, s)):
            raise ValueError(f'root should contain only scene folders, but got file {s}')

        kinds = sorted(os.listdir(os.path.join(cfg.root, s)))
        for kind in kinds:
            kind_path = os.path.join(cfg.root, s, kind)
            print(kind_path)
            p_folders = sorted(os.listdir(kind_path))
            for p_folder in p_folders:
                p_path = os.path.join(kind_path, p_folder)
                if not os.path.isdir(p_path):
                    continue
                root_left = os.path.join(cfg.root, s, kind, p_folder, 'image_left')
                root_right = os.path.join(cfg.root, s, kind, p_folder, 'image_right')
                out = os.path.join(cfg.out, s, kind, p_folder)
                print('out path: ', out, root_left, root_right)
                run_simulation_LCD(
                    root_left,
                    root_right,
                    out,
                    crop_size,
                    cfg.random_crop,
                    cfg.threshold,
                    cfg.length,
                    cfg.noise,
                    cfg.noise_level,
                    cfg.save_spike_sequences,
                )


def parse_args(argv=None):
    return parse_simulation_args(
        argv,
        default_config='tartanair_simulation.yaml',
        default_root='data/tartanair/interp',
        default_out='data/tartanair/sim',
        default_crop='256x256',
        description='Tartanair spike-simulation preprocessing pipeline.',
    )


def main(argv=None):
    args = parse_args(argv)
    cfg = namespace_to_simulation_config(args, config_cls=Config)
    entry(cfg)


if __name__ == '__main__':
    main()
