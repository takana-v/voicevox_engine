import sys
import traceback
from typing import Dict, List

from .synthesis_engine_espnet import SynthesisEngineBase, SynthesisEngineEspnet


def make_synthesis_engines(
    speakers_settings: List,
    use_gpu: bool,
    enable_mock: bool = True,
) -> Dict[str, SynthesisEngineBase]:
    """
    音声ライブラリをロードして、音声合成エンジンを生成

    Parameters
    ----------
    speakers_setting: List
        speakersにfrom_pretrained_argsを加えたもの
        run.py参照のこと
    use_gpu: bool
        音声ライブラリに GPU を使わせるか否か
    enable_mock: bool, optional, default=True
        コア読み込みに失敗したときにエラーを送出するかどうか
        Falseだと代わりにmockが使用される
    """
    engine_version = speakers_settings[0]["version"]
    synthesis_engines = {}
    try:
        synthesis_engines[engine_version] = SynthesisEngineEspnet(
            speakers_settings=speakers_settings, use_gpu=use_gpu
        )
    except Exception:
        if not enable_mock:
            raise
        traceback.print_exc()
        print(
            "Notice: mock-library will be used.",
            file=sys.stderr,
        )
        from ..dev.core import metas as mock_metas
        from ..dev.core import supported_devices as mock_supported_devices
        from ..dev.synthesis_engine import MockSynthesisEngine

        if "0.0.0" not in synthesis_engines:
            synthesis_engines["0.0.0"] = MockSynthesisEngine(
                speakers=mock_metas(), supported_devices=mock_supported_devices()
            )

    return synthesis_engines
