import os
import shutil
import sys
from multiprocessing import Lock, Pipe, Process
from multiprocessing.connection import Connection
from pathlib import Path

from .synthesis_engine import SynthesisEngine, SynthesisEngineBase


def engine_process(
    use_gpu: bool,
    old_voicelib_dir: Path,
    libtorch_dir: Path,
    conn: Connection,
):
    try:
        if sys.platform == "win32":
            os.add_dll_directory(str(libtorch_dir.resolve()))
            os.add_dll_directory(str(old_voicelib_dir.resolve()))
        elif sys.platform == "linux" or sys.platform == "darwin":
            os.environ["LD_LIBRARY_PATH"] = (
                str(libtorch_dir.resolve()) + ":" + str(old_voicelib_dir.resolve())
            )
        else:
            raise RuntimeError("Unsupported OS")
        sys.path.insert(0, str(old_voicelib_dir.resolve()))
        import core

        core.initialize(str(old_voicelib_dir.resolve()), use_gpu)
        synthesis_engine = SynthesisEngine(
            yukarin_s_forwarder=core.yukarin_s_forward,
            yukarin_sa_forwarder=core.yukarin_sa_forward,
            decode_forwarder=core.decode_forward,
            speakers=core.metas(),
        )
    except Exception as e:
        conn.send(e)
        sys.exit(1)
    while True:
        try:
            data = conn.recv()
        except EOFError:
            # パイプが破棄された場合
            sys.exit(1)
        try:
            if data[0] == "speakers":
                conn.send(synthesis_engine.speakers)
            elif data[0] == "default_sampling_rate":
                conn.send(synthesis_engine.default_sampling_rate)
            else:
                conn.send(eval(f"synthesis_engine.{data[0]}(*data[1], **data[2])"))
        except Exception as e:
            try:
                conn.send(e)
            except Exception:
                sys.exit(1)


class OldCoreEngine(SynthesisEngineBase):
    def __init__(
        self,
        use_gpu: bool,
        old_voicelib_dir: Path,
        libtorch_dir: Path,
    ):
        conn1, conn2 = Pipe()
        self.process = Process(
            target=engine_process,
            args=(use_gpu, old_voicelib_dir, libtorch_dir, conn2),
            daemon=True,
        )
        self.process.start()
        self.conn = conn1
        self.lock = Lock()
        # 疎通確認
        self.conn.send(["speakers", (), {}])
        while True:
            if self.conn.poll(0.1):
                ret = self.conn.recv()
                if "__traceback__" in dir(ret):
                    raise ret
                else:
                    break
            else:
                if not self.process.is_alive():
                    raise RuntimeError("過去エンジンの起動に失敗しました。")

    def _recv_data(self, fc_name, args, kwargs):
        if self.process.is_alive():
            with self.lock:
                self.conn.send([fc_name, args, kwargs])
                ret = self.conn.recv()
                if "__traceback__" in dir(ret):
                    raise ret
                else:
                    return ret
        else:
            raise ConnectionRefusedError("エンジンが既に終了しています。")

    @property
    def speakers(self):
        return self._recv_data("speakers", (), {})

    @property
    def default_sampling_rate(self):
        return self._recv_data("default_sampling_rate", (), {})

    def replace_phoneme_length(self, *args, **kwargs):
        return self._recv_data("replace_phoneme_length", args, kwargs)

    def replace_mora_pitch(self, *args, **kwargs):
        return self._recv_data("replace_mora_pitch", args, kwargs)

    def replace_mora_data(self, *args, **kwargs):
        return self._recv_data("replace_mora_data", args, kwargs)

    def create_accent_phrases(self, *args, **kwargs):
        return self._recv_data("create_accent_phrases", args, kwargs)

    def synthesis(self, *args, **kwargs):
        return self._recv_data("synthesis", args, kwargs)


def make_old_core_engine(
    use_gpu: bool,
    old_voicelib_dir: Path,
    voicevox_dir: Path,
    libtorch_dir: Path,
) -> OldCoreEngine:
    if old_voicelib_dir is None or libtorch_dir is None:
        raise RuntimeError("old_voicelib_dirとlibtorch_dirの指定が必要です")
    if voicevox_dir is None:
        voicevox_dir = Path("")
    if not (old_voicelib_dir / "core.pyd").is_file():
        if not (voicevox_dir / "core.pyd").is_file():
            raise RuntimeError("コアモジュールのファイルが見つかりません")
        shutil.copyfile(
            str(voicevox_dir / "core.pyd"), str(old_voicelib_dir / "core.pyd")
        )
    return OldCoreEngine(use_gpu, old_voicelib_dir, libtorch_dir)