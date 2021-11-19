import os
import zipfile
from typing import List, NamedTuple

import gdown
import numpy as np
import pyopenjtalk
import resampy
import sklearn.neighbors._partition_nodes
import torch
import yaml
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import load_model
from parallel_wavegan.utils import read_hdf5
from sklearn.preprocessing import StandardScaler

from voicevox_engine.model import AccentPhrase, AudioQuery


class EspnetSettings(NamedTuple):
    acoustic_model_config_path: str
    acoustic_model_path: str
    vocoder_model_path: str
    vocoder_stats_path: str


class EspnetModel:
    def __init__(self, settings: EspnetSettings, use_gpu=False, use_scaler=False):
        # init run for pyopenjtalk
        pyopenjtalk.g2p('a')

        device = 'cuda' if use_gpu else 'cpu'
        self.acoustic_model = Text2Speech(
            settings.acoustic_model_config_path,
            settings.acoustic_model_path,
            device=device,
            threshold=0.5,
            minlenratio=0.0,
            maxlenratio=10.0,
            use_att_constraint=False,
            backward_window=1,
            forward_window=3
        )
        self.acoustic_model.spc2wav = None
        self.vocoder = load_model(settings.vocoder_model_path).to(device).eval()

        self.use_scaler = use_scaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = read_hdf5(settings.vocoder_stats_path, "mean")
        self.scaler.scale_ = read_hdf5(settings.vocoder_stats_path, "scale")
        self.scaler.n_features_in_ = self.scaler.mean_.shape[0]

        with open(settings.acoustic_model_config_path) as f:
            config = yaml.safe_load(f)
        self.token_id_converter = TokenIDConverter(
            token_list=config["token_list"],
            unk_symbol="<unk>",
        )

    def make_voice(self, tokens, seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        text_ints = np.array(self.token_id_converter.tokens2ids(tokens), dtype=np.int64)
        with torch.no_grad():
            output = self.acoustic_model(text_ints)
            if self.use_scaler:
                mel = self.scaler.transform(output['feat_gen_denorm'])
                wave = self.vocoder.inference(mel)
            else:
                wave = self.vocoder.inference(output['feat_gen'])
        wave = wave.view(-1).cpu().numpy()
        return wave

    @classmethod
    def get_tsukuyomichan_model(cls, use_gpu):
        download_path = './models'
        model_path = f"{download_path}/TSUKUYOMICHAN_COEIROINK_MODEL_v.2.0.0"
        model_url = 'https://drive.google.com/uc?id=1jPuUoWoGc231ilNzA647tN4EQthl-BU7'
        acoustic_model_path = f"{model_path}/ACOUSTIC_MODEL/100epoch.pth"
        acoustic_model_config_path = f"{model_path}/ACOUSTIC_MODEL/config.yaml"
        acoustic_model_stats_path = f"{model_path}/ACOUSTIC_MODEL/feats_stats.npz"
        vocoder_model_path = f"{model_path}/VOCODER/checkpoint-2500000steps.pkl"
        vocoder_stats_path = f"{model_path}/VOCODER/stats.h5"
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        if not os.path.exists(model_path):
            cls.download_model(download_path, model_path, model_url)
            cls.update_acoustic_model_config(acoustic_model_config_path, acoustic_model_stats_path)

        settings = EspnetSettings(
            acoustic_model_config_path=acoustic_model_config_path,
            acoustic_model_path=acoustic_model_path,
            vocoder_model_path=vocoder_model_path,
            vocoder_stats_path=vocoder_stats_path
        )
        return cls(settings, use_gpu=use_gpu, use_scaler=True)

    @staticmethod
    def download_model(download_path, model_path, model_url):
        zip_path = f"{model_path}.zip"
        gdown.download(model_url, zip_path, quiet=False)
        with zipfile.ZipFile(zip_path) as model_zip:
            model_zip.extractall(download_path)
        os.remove(zip_path)

    @staticmethod
    def update_acoustic_model_config(acoustic_model_config_path, acoustic_model_stats_path):
        with open(acoustic_model_config_path) as f:
            yml = yaml.safe_load(f)
        if not yml['normalize_conf']['stats_file'] == acoustic_model_stats_path:
            yml['normalize_conf']['stats_file'] = acoustic_model_stats_path
            with open(acoustic_model_config_path, 'w') as f:
                yaml.safe_dump(yml, f)
            print("Update acoustic model yaml.")


class SynthesisEngine:
    def __init__(self, **kwargs):
        self.speakers = kwargs["speakers"]

        self.default_sampling_rate = 24000
        self.use_gpu = False

        self.speaker_models: List[EspnetModel] = []
        self.speaker_models.append(EspnetModel.get_tsukuyomichan_model(use_gpu=self.use_gpu))

    @staticmethod
    def replace_phoneme_length(accent_phrases: List[AccentPhrase], speaker_id: int) -> List[AccentPhrase]:
        return accent_phrases

    @staticmethod
    def replace_mora_pitch(accent_phrases: List[AccentPhrase], speaker_id: int) -> List[AccentPhrase]:
        return accent_phrases

    def synthesis(self, query: AudioQuery, speaker_id: int, text: str = '') -> np.ndarray:
        # make_wave
        tokens = self.query2tokens_prosody(query, text)
        wave = self.speaker_models[speaker_id].make_voice(tokens)

        # trim
        wave = self.trim_wav_start_and_end_sil(wave)

        # volume
        if query.volumeScale != 1:
            wave *= query.volumeScale

        if query.prePhonemeLength != 0 or query.postPhonemeLength != 0:
            pre_pause = np.zeros(int(self.default_sampling_rate * query.prePhonemeLength))
            post_pause = np.zeros(int(self.default_sampling_rate * query.postPhonemeLength))
            wave = np.concatenate([pre_pause, wave, post_pause], 0)

        # resampling
        if query.outputSamplingRate != self.default_sampling_rate:
            wave = resampy.resample(
                wave,
                self.default_sampling_rate,
                query.outputSamplingRate,
                filter="kaiser_fast",
            )

        return wave

    @staticmethod
    def query2tokens_with_pause_and_accent(query: AudioQuery):
        tokens = []
        for accent_phrase in query.accent_phrases:
            for i, mora in enumerate(accent_phrase.moras):
                if mora.consonant:
                    tokens.append(mora.consonant)
                    tokens.append(str(accent_phrase.accent))
                    tokens.append(str(i - (accent_phrase.accent - 1)))
                tokens.append(mora.vowel)
                tokens.append(str(accent_phrase.accent))
                tokens.append(str(i - (accent_phrase.accent - 1)))
            if accent_phrase.pause_mora:
                tokens.append('pau')
        tokens.append('<sos/eos>')
        return tokens

    @staticmethod
    def query2tokens_prosody(query: AudioQuery, text=''):
        question_flag = False
        if query.kana != '':
            if query.kana[-1] in ['?', '？']:
                question_flag = True
        if text != '':
            if text[-1] in ['?', '？']:
                question_flag = True
        tokens = ['^']
        for i, accent_phrase in enumerate(query.accent_phrases):
            up_token_flag = False
            for j, mora in enumerate(accent_phrase.moras):
                if mora.consonant:
                    tokens.append(mora.consonant.lower())
                if mora.vowel == 'N':
                    tokens.append(mora.vowel)
                else:
                    tokens.append(mora.vowel.lower())
                if accent_phrase.accent == j+1 and j+1 != len(accent_phrase.moras):
                    tokens.append(']')
                if accent_phrase.accent-1 >= j+1 and up_token_flag is False:
                    tokens.append('[')
                    up_token_flag = True
            if i+1 != len(query.accent_phrases):
                if accent_phrase.pause_mora:
                    tokens.append('_')
                else:
                    tokens.append('#')
        if question_flag:
            tokens.append('?')
        else:
            tokens.append('$')
        return tokens

    @staticmethod
    def trim_wav_start_and_end_sil(wave, threshold=0.001):
        start = 0
        for i, w in enumerate(wave):
            if abs(w) >= threshold:
                start = i
                break
        wave = wave[start:]
        end = 0
        wave = np.flipud(wave)
        for i, w in enumerate(wave):
            if abs(w) >= 0.001:
                end = i
                break
        wave = wave[end:]
        wave = np.flipud(wave)
        return wave
