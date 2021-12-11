from pathlib import Path
from typing import Optional

from .coeiroink_engine import SynthesisEngine, SynthesisEngineBase


def make_synthesis_engine(
    use_gpu: bool,
    voicelib_dir: Path,
    voicevox_dir: Optional[Path] = None,
) -> SynthesisEngineBase:
    """
    音声ライブラリをロードして、音声合成エンジンを生成

    Parameters
    ----------
    use_gpu: bool
        音声ライブラリに GPU を使わせるか否か
    voicelib_dir: Path
        音声ライブラリ自体があるディレクトリ
    voicevox_dir: Path, optional, default=None
        音声ライブラリの Python モジュールがあるディレクトリ
        None のとき、Python 標準のモジュール検索パスのどれかにあるとする
    """
    # todo: speakers修正
    return SynthesisEngine(speakers="")
