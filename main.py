from mmsf import MultimodalSlowFast
from asf.audio_slowfast.tools.run_net import load_config


def main() -> None:
    cfg = load_config()
    model = MultimodalSlowFast(cfg=cfg)


if __name__ == "__main__":
    main()
