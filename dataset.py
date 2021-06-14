from torch.utils.data import Dataset
import os
from typing import Tuple, Union, List
from torch import Tensor
import torchaudio
import torch.nn.functional as F
import torch
import librosa
import numpy as np
from torchaudio.transforms import MelSpectrogram, Resample
SampleType = Tuple[Tensor, int, str, str, str]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def pad(x, m_len):
  out = F.pad(x, (0, m_len-x.size(-1)))
  return out

def segment(x, seglen=128, r=None, return_r=False):
    if x.shape[1] < seglen:
        y = pad(x, seglen)
    elif x.shape[1] == seglen:
        y = x
    else:
        if r is None:
            r = np.random.randint(x.shape[1] - seglen)
        y = x[:,r:r+seglen]
    if return_r:
        return y, r
    else:
        return y


def collate_fn(tensors):
  for i in range(len(tensors)):
    #tensors[i] = segment(tensors[i]).unsqueeze(0)
    tensors[i] = segment(tensors[i])
  return {'features':torch.cat(tensors).to(device)}




#https://github.com/pytorch/audio/blob/master/torchaudio/datasets/vctk.py

class VCTK_092_CUSTOM(Dataset):
    """Create VCTK 0.92 Dataset
    Args:
        root (str): Root directory where the dataset's top level directory is found.
        mic_id (str): Microphone ID. Either ``"mic1"`` or ``"mic2"``. (default: ``"mic2"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        url (str, optional): The URL to download the dataset from.
            (default: ``"https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"``)
        audio_ext (str, optional): Custom audio extension if dataset is converted to non-default audio format.
    Note:
        * All the speeches from speaker ``p315`` will be skipped due to the lack of the corresponding text files.
        * All the speeches from ``p280`` will be skipped for ``mic_id="mic2"`` due to the lack of the audio files.
        * Some of the speeches from speaker ``p362`` will be skipped due to the lack of  the audio files.
        * See Also: https://datashare.is.ed.ac.uk/handle/10283/3443
    """

    def __init__(
            self,
            root: str,
            speaker_ids: list,
            utterance_count: int = 200,
            mic_id: str = "mic2",
            audio_ext=".flac",
    ):
        if mic_id not in ["mic1", "mic2"]:
            raise RuntimeError(
                f'`mic_id` has to be either "mic1" or "mic2". Found: {mic_id}'
            )

        self._path = os.path.join(root, "VCTK-Corpus-0.92")
        self._txt_dir = os.path.join(self._path, "txt")
        self._audio_dir = os.path.join(self._path, "wav48_silence_trimmed")
        self._mic_id = mic_id
        self._audio_ext = audio_ext

        # Extracting speaker IDs from the folder structure
        self._speaker_ids = speaker_ids
        self._sample_ids = []

        """
        Due to some insufficient data complexity in the 0.92 version of this dataset,
        we start traversing the audio folder structure in accordance with the text folder.
        As some of the audio files are missing of either ``mic_1`` or ``mic_2`` but the
        text is present for the same, we first check for the existence of the audio file
        before adding it to the ``sample_ids`` list.
        Once the ``audio_ids`` are loaded into memory we can quickly access the list for
        different parameters required by the user.
        """
        for speaker_id in self._speaker_ids:
            if speaker_id == "p280" and mic_id == "mic2":
                continue
            utterance_dir = os.path.join(self._txt_dir, speaker_id)
            for utterance_file in sorted(
                    f for f in os.listdir(utterance_dir) if f.endswith(".txt")
            )[:utterance_count]:
                utterance_id = os.path.splitext(utterance_file)[0]
                audio_path_mic = os.path.join(
                    self._audio_dir,
                    speaker_id,
                    f"{utterance_id}_{mic_id}{self._audio_ext}",
                )
                if speaker_id == "p362" and not os.path.isfile(audio_path_mic):
                    continue
                self._sample_ids.append(utterance_id.split("_"))

    def _load_audio(self, file_path):
        data, sample_rate = torchaudio.load(file_path)
        #data, sample_rat = librosa.load(file_path)
        mel = torchaudio.transforms.Resample(sample_rate,22050)(data)
        mel = torchaudio.transforms.MelSpectrogram(22050, 1024, 1024, 256, n_mels=80)(mel)
        
        #mel = librosa.feature.melspectrogram(data, sr=22050,n_mels=80, win_length=1024,
        #                                     hop_length=256)
        return mel

    def _load_sample(self, speaker_id: str, utterance_id: str, mic_id: str) -> SampleType:
        utterance_path = os.path.join(
            self._txt_dir, speaker_id, f"{speaker_id}_{utterance_id}.txt"
        )
        audio_path = os.path.join(
            self._audio_dir,
            speaker_id,
            f"{speaker_id}_{utterance_id}_{mic_id}{self._audio_ext}",
        )
        # Reading FLAC
        #waveform, sample_rate = self._load_audio(audio_path)
        mel = self._load_audio(audio_path)

        #return (waveform, sample_rate, utterance, speaker_id, utterance_id)
        #return [mel, speaker_id, utterance_id]
        return torch.tensor(mel).log10().float()

    def __getitem__(self, n: int) -> SampleType:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the sample to be loaded
        Returns:
            tensor: ``mel``
        """
        speaker_id, utterance_id = self._sample_ids[n]
        return self._load_sample(speaker_id, utterance_id, self._mic_id)

    def __len__(self) -> int:
        return len(self._sample_ids)