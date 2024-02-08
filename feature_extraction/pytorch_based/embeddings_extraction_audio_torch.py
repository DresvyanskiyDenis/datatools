import glob
import os
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoProcessor, HubertModel, ASTModel
from scipy.io import wavfile
import scipy.signal as sps


def resample_audio(audio:np.ndarray, old_frame_rate:int, new_frame_rate:int)->np.ndarray:
    if old_frame_rate == new_frame_rate:
        return audio
    # Resample data
    number_of_samples = round(len(audio) * float(new_frame_rate) / old_frame_rate)
    resampled = sps.resample(audio, number_of_samples)
    return resampled



class AudioEmbeddingsExtractor:
    def __init__(self, extractor_type:str, frame_rate:int):
        # checking params
        if extractor_type not in ('wav2vec', 'HuBERT', 'AudioSpectrogramTransformer'):
            raise ValueError("Invalid extractor_type: {}".format(extractor_type))
        if frame_rate <= 0:
            raise ValueError("Invalid frame_rate: {}".format(frame_rate))
        self.extractor_type = extractor_type
        self.frame_rate = frame_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__initialize()
        self.model.to(self.device)
        print(f"{self.extractor_type} Embeddings extractor has been initialized to the device: {self.device}.")

        self.embeddings_size = {
            'wav2vec': 768,
            'HuBERT': 1019,
            'AudioSpectrogramTransformer': 768,
        }


    def get_embeddings_size(self):
        return self.embeddings_size[self.extractor_type]

    def __initialize_wav2vec(self):
        model_name = "facebook/wav2vec2-base-960h" # many variants are possible, see:
        # https://huggingface.co/models?other=wav2vec2&sort=trending&search=facebook%2Fwav2vec2
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)

    def __initialize_HuBERT(self):
        model_name = "facebook/hubert-large-ls960-ft"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name)
        # there is workaround with other lib, see: torchaudio bundle

    def __initialize_AudioSpectrogramTransformer(self):
        model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = ASTModel.from_pretrained(model_name)

    def __initialize(self):
        if self.extractor_type == 'wav2vec':
            self.__initialize_wav2vec()
        elif self.extractor_type == 'HuBERT':
            self.__initialize_HuBERT()
        elif self.extractor_type == 'AudioSpectrogramTransformer':
            self.__initialize_AudioSpectrogramTransformer()
        else:
            raise ValueError("Invalid extractor_type: {}".format(self.extractor_type))


    def __extract_features_wav2vec2(self, audio:torch.Tensor)->np.ndarray:
        with torch.no_grad():
            features = self.model(audio)
            # get last hidden state
            features = features.last_hidden_state
            # average features
            features = torch.mean(features, dim=1)
            features = features.cpu().detach().numpy().squeeze()
        return features

    def __extract_features_HuBERT(self, audio:torch.Tensor)->np.ndarray:
        with torch.no_grad():
            features = self.model(audio)
            # get last hidden state
            features = features.last_hidden_state
            # average features
            features = torch.mean(features, dim=1)
            features = features.cpu().detach().numpy().squeeze()
        return features

    def __extract_features_AudioSpectrogramTransformer(self, audio:torch.Tensor)->np.ndarray:
        with torch.no_grad():
            features = self.model(audio)
            # get last hidden state
            features = features.last_hidden_state
            # average features
            features = torch.mean(features, dim=1)
            features = features.cpu().detach().numpy().squeeze()
        return features


    def _extract_features(self, audio:Union[str,
                                  Tuple[np.ndarray, int],
                                  Tuple[torch.Tensor, int]
    ])->np.ndarray:
        if isinstance(audio, str):
            sr, audio = wavfile.read(audio)
        elif isinstance(audio, tuple):
            audio, sr = audio
            if isinstance(audio, torch.Tensory):
                audio = audio.numpy()
        else:
            raise ValueError("Invalid audio format: {}. Should be either str, numpy, or torch.Tensor".format(audio))


        if sr != self.frame_rate:
            audio = resample_audio(audio, sr, self.frame_rate)
            sr = self.frame_rate

        if len(audio.shape)==2 and audio.shape[-1]>1:
            print("Warning. Detected more than one channel in audio. Averaging channels.")
            audio = np.mean(audio, axis=-1)

        # check if there is no channels dimension
        if len(audio.shape)!=1:
            raise ValueError("Invalid audio shape: {}. Should be 1-dimensional".format(audio.shape))

        # transform audio to float32
        audio = audio.astype("float32")

        preprocessed = self.processor(audio, sampling_rate=self.frame_rate, return_tensors="pt")
        preprocessed = preprocessed.to(self.device)
        preprocessed.input_values = preprocessed.input_values.float()

        if self.extractor_type == 'wav2vec':
            features = self.__extract_features_wav2vec2(preprocessed.input_values)
        elif self.extractor_type == 'HuBERT':
            features = self.__extract_features_HuBERT(preprocessed.input_values)
        elif self.extractor_type == 'AudioSpectrogramTransformer':
            features = self.__extract_features_AudioSpectrogramTransformer(preprocessed.input_values)

        return features

    def extract_features_dir(self, path_to_data: str, labels: pd.DataFrame,
                             first_n_rows: Union[None, int] = None) -> pd.DataFrame:
        # labels format: [filename, category, arousal, valence, dominance]
        # important: filename should be only the filename, not the full path
        # Output dataframe format: [filename, category, arousal, valence, dominance, x_0, x_1, ..., x_n,]

        # infer hom much features the model outputs
        tmp = self._extract_features(os.path.join(path_to_data, labels.iloc[0,0]))
        num_features = tmp.shape[-1]
        # create output dataframe
        output = pd.DataFrame(columns=['filename'] + ['category', 'arousal', 'valence', 'dominance']
                                      + ['x_{}'.format(i) for i in range(num_features)])
        if first_n_rows is not None:
            labels = labels.iloc[0:first_n_rows, :]
        # iterate over files
        for row_idx, row in tqdm(labels.iterrows(), total=labels.shape[0]):
            filename = row.filename
            category = row.category
            arousal = row.arousal
            valence = row.valence
            dominance = row.dominance
            # extract features
            features = self._extract_features(os.path.join(path_to_data, filename))
            # add to dataframe
            new_row = pd.DataFrame.from_dict({'filename':[filename],
                                    'category':[category],
                                    'arousal':[arousal],
                                    'valence':[valence],
                                    'dominance':[dominance],
                                    **{'x_{}'.format(i):[features[i]] for i in range(num_features)}
                                    })
            output = pd.concat([output, new_row], ignore_index=True)
        return output


    def extract_features_audio(self, audio:Union[str, Tuple[np.ndarray, int]], chunk_size:float)->List[np.ndarray]:
        '''
        Extract features from provided audio, cutting it on chunks beforehand
        :param audio: Union[str, Tuple[np.ndarray, int]]
            THe audio to extract features from. Can be either path to the audio file, or tuple of (audio, frame_rate)
        :param chunk_size: float
            The size of the chunk in seconds
        :return: List[np.ndarray]
            List of extracted features, every element represents features of one chunk of provided length
        '''
        if isinstance(audio, str):
            sr, audio = wavfile.read(audio)
        else:
            audio, sr = audio

        if sr != self.frame_rate:
            audio = resample_audio(audio, sr, self.frame_rate)
            sr = self.frame_rate

        if len(audio.shape)==2 and audio.shape[-1]>1:
            print("Warning. Detected more than one channel in audio. Averaging channels.")
            audio = np.mean(audio, axis=-1)

        # cut audio into chunks
        chunk_size = int(chunk_size*sr)
        chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
        # change the last chunk so that it has the same size as others
        chunks[-1] = audio[-chunk_size:]

        # transform audio to float32
        chunks = [chunk.astype("float32") for chunk in chunks]

        # extract features
        extracted_features = []
        for idx, chunk in enumerate(chunks):
            preprocessed_chunk = self.processor(chunk, sampling_rate=self.frame_rate, return_tensors="pt")
            preprocessed_chunk = preprocessed_chunk.to(self.device)
            preprocessed_chunk.input_values = preprocessed_chunk.input_values.float()
            if self.extractor_type == 'wav2vec':
                features = self.__extract_features_wav2vec2(preprocessed_chunk.input_values)
            elif self.extractor_type == 'HuBERT':
                features = self.__extract_features_HuBERT(preprocessed_chunk.input_values)
            elif self.extractor_type == 'AudioSpectrogramTransformer':
                features = self.__extract_features_AudioSpectrogramTransformer(preprocessed_chunk.input_values)
            extracted_features.append(features)

        del chunks
        return extracted_features


if __name__=="__main__":
    extractor = AudioEmbeddingsExtractor(extractor_type='AudioSpectrogramTransformer', frame_rate=16000)
    audio = np.random.randint(-32768, 32767, 16000*5)
    features = extractor.extract_features_audio((audio, 16000), chunk_size=1)