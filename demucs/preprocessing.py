import torch as th 
import torch.nn.functional as F
import torchaudio
import os
from tqdm import tqdm

class PreprocessSTFT:
    def __init__(self, fft_size =1024, hop_size = 512, win_length = 1024, window = "hann_window"):
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = getattr(th, window)(win_length)
    
    def pad_audio(self, audio):
        """Pad audio to the maximum length.

        Args:
            audio (Tensor): Input audio tensor (C, T).

        Returns:
            Tensor: Padded audio tensor (C, max_length).
        """
        max_length = self.win_length
        if audio.size(-1) < max_length:
            audio = F.pad(audio, (0, max_length - audio.size(-1)))
        return audio
    
    def compute_stft(self, audio):
        audio = self.pad_audio(audio)
        x_stft = th.stft(audio, self.fft_size, self.hop_size, self.win_length, window=self.window, return_complex=True)
        return x_stft
    
    def preprocess_and_save(self, audio_files, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for audio_file in audio_files:
            audio, sr = torchaudio.load(audio_file)
            x_stft = self.compute_stft(audio)
            output_file = os.path.join(output_dir, os.path.basename(audio_file).replace(".wav", ".pt"))
            th.save(x_stft, output_file)


def preprocess_all_files(input_dir, output_dir):
    for root, dirs, files in tqdm(os.walk(input_dir)):
        for file in files:
            if file.endswith(".wav"):
                audio_files = [os.path.join(root, file)]
                preprocess = PreprocessSTFT()
                preprocess.preprocess_and_save(audio_files, os.path.join(output_dir, root))

preprocess_all_files("/mnt/data/mads/generatedFilesSubsetSTFT/train", "/mnt/data/mads/generatedFilesSubsetSTFT/trainSTFT")
