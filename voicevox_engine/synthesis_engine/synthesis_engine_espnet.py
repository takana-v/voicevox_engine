import json
from typing import List, Optional

import numpy as np
from espnet2.bin.tts_inference import Text2Speech
from espnet2.text.token_id_converter import TokenIDConverter
from fastapi import HTTPException
from scipy.signal import resample

from ..model import AccentPhrase, AudioQuery, Speaker, SpeakerStyle
from .synthesis_engine_base import SynthesisEngineBase


def audio_query2int_tensor(
    audio_query: AudioQuery, token_id_converter: TokenIDConverter
):
    tokens = []
    for accent_phrase in audio_query.accent_phrases:
        accent = accent_phrase.accent
        for i, mora in enumerate(accent_phrase.moras):
            if mora.consonant is not None:
                tokens.append(mora.consonant)
                tokens.append(str(accent))
                tokens.append(str((i + 1) - accent))
            tokens.append(mora.vowel)
            tokens.append(str(accent))
            tokens.append(str((i + 1) - accent))
        if accent_phrase.pause_mora is not None:
            tokens.append(accent_phrase.pause_mora.vowel)
    return np.array(token_id_converter.tokens2ids(tokens))


class SynthesisEngineEspnet(SynthesisEngineBase):
    def __init__(self, speakers_settings, use_gpu):
        self.speakers_settings = speakers_settings
        self.default_sampling_rate = 44100
        for speaker in self.speakers_settings:
            speaker["text2speech"] = Text2Speech.from_pretrained(
                **speaker["text2speech_args"]
            )
            speaker["token_id_converter"] = TokenIDConverter(
                **speaker["token_id_converter_args"]
            )

    @property
    def speakers(self) -> str:
        return json.dumps(
            [
                Speaker(
                    name=speaker["name"],
                    speaker_uuid=speaker["speaker_uuid"],
                    styles=[
                        SpeakerStyle(name=style["name"], id=style["id"])
                        for style in speaker["styles"]
                    ],
                    version=speaker["version"],
                ).dict()
                for speaker in self.speakers_settings
            ]
        )

    @property
    def supported_devices(self) -> Optional[str]:
        return None

    def replace_phoneme_length(
        self, accent_phrases: List[AccentPhrase], speaker_id: int
    ) -> List[AccentPhrase]:
        """
        accent_phrasesの母音・子音の長さを設定する
        Parameters
        ----------
        accent_phrases : List[AccentPhrase]
            アクセント句モデルのリスト
        speaker_id : int
            話者ID
        Returns
        -------
        accent_phrases : List[AccentPhrase]
            母音・子音の長さが設定されたアクセント句モデルのリスト
        """
        # 母音・子音の長さを設定するのは不可能なのでそのまま返す
        return accent_phrases

    def replace_mora_pitch(
        self, accent_phrases: List[AccentPhrase], speaker_id: int
    ) -> List[AccentPhrase]:
        """
        accent_phrasesの音高(ピッチ)を設定する
        Parameters
        ----------
        accent_phrases : List[AccentPhrase]
            アクセント句モデルのリスト
        speaker_id : int
            話者ID
        Returns
        -------
        accent_phrases : List[AccentPhrase]
            音高(ピッチ)が設定されたアクセント句モデルのリスト
        """
        # 音高を設定するのは不可能なのでそのまま返す
        return accent_phrases

    def _synthesis_impl(self, query: AudioQuery, speaker_id: int):
        """
        音声合成クエリから音声合成に必要な情報を構成し、実際に音声合成を行う
        Parameters
        ----------
        query : AudioQuery
            音声合成クエリ
        speaker_id : int
            話者ID
        Returns
        -------
        wave : numpy.ndarray
            音声合成結果
        """
        for speaker in self.speakers_settings:
            if speaker["styles"][0]["id"] == speaker_id:
                text2speech = speaker["text2speech"]
                token_id_converter = speaker["token_id_converter"]
                text2speech_call_args = speaker["text2speech_call_args"]
                break
        else:
            raise HTTPException(status_code=422, detail="該当する話者が見つかりません")
        int_tensor = audio_query2int_tensor(query, token_id_converter)
        wave = (
            text2speech(int_tensor, **text2speech_call_args)["wav"]
            .view(-1)
            .cpu()
            .numpy()
        )
        return resample(
            wave,
            24000 * len(wave) // self.default_sampling_rate,
        )
