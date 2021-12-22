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
    abs_libtorch_dir = str(libtorch_dir.resolve())
    abs_old_voicelib_dir = str(old_voicelib_dir.resolve())
    try:
        if sys.platform == "win32":
            os.add_dll_directory(abs_libtorch_dir)
            os.add_dll_directory(abs_old_voicelib_dir)
        elif sys.platform == "linux" or sys.platform == "darwin":
            os.environ["LD_LIBRARY_PATH"] = (
                abs_libtorch_dir + ":" + abs_old_voicelib_dir
            )
        else:
            raise RuntimeError("Unsupported OS")
        sys.path.insert(0, abs_old_voicelib_dir)
        import core

        core.initialize(abs_old_voicelib_dir, use_gpu)
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
    if sys.platform == "win32":
        core_module_name = "core.pyd"
    elif sys.platform == "linux":
        core_module_name = "core.so"
    elif sys.platform == "darwin":
        core_module_name = "core.dylib" # not tested
    if old_voicelib_dir is None or libtorch_dir is None:
        raise RuntimeError("old_voicelib_dirとlibtorch_dirの指定が必要です")
    if voicevox_dir is None:
        voicevox_dir = Path("")
    if not (old_voicelib_dir / core_module_name).is_file():
        if not (voicevox_dir / core_module_name).is_file():
            raise RuntimeError("コアモジュールのファイルが見つかりません")
        shutil.copyfile(
            str(voicevox_dir / core_module_name), str(old_voicelib_dir / core_module_name)
        )
    return OldCoreEngine(use_gpu, old_voicelib_dir, libtorch_dir)
