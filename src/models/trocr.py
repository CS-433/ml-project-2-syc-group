from itertools import groupby
import typing
import torch 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import transformers 
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from mltu.torch.model import Model, ModelEMA  
from mltu.transformers import LabelIndexer 
from mltu.torch.metrics import Metric 
from mltu.utils.text_utils import get_cer, get_wer



class TrOCRWrapper(nn.Module): 
    def __init__(self, model_path: str, device):
        super(TrOCRWrapper, self).__init__()
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model.to(device)

        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id

        # set special tokens used for creating the decoder_input_ids from the labels
        #self.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        # make sure vocab size is set correctly
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        # set beam search parameters
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.model.config.max_length = 8
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4
        
        
    def forward(self, image, label ): 
        # import matplotlib.pyplot as plt 
        
    
        
        # print("image device", image.device)
        # image = image.permute(0, 3, 1, 2)
        # print("image shape", image.shape)
        # print("")
        # print("label", label)
        #image = self.processor(images=image, return_tensors="pt").pixel_values
        #   image = image.to("mps")
        # print("hi")
        # input_id to label
        #img_label = self.processor.tokenizer.decode(label[0]) 
        # print(label[0].max())
        #label[label == -100] = self.processor.tokenizer.pad_token_id
        # print("Max token ID allowed:", self.processor.tokenizer.vocab_size - 1)
        
        # plt.title(genera©≈bfgn h
        outputs = self.model(pixel_values=image, labels = label)
        # print("outputs", outputs) 
        return outputs 

    




class ModelTrOCR(Model): 
    
    def train_step(
        self, 
        data: typing.Union[np.ndarray, torch.Tensor], 
        target: typing.Union[np.ndarray, torch.Tensor],
        loss_info: dict = {}
    ) -> torch.Tensor:
        """ Perform one training step

        Args:
            data (typing.Union[np.ndarray, torch.Tensor]): training data
            target (typing.Union[np.ndarray, torch.Tensor]): training target
            loss_info (dict, optional): additional loss information. Defaults to {}.

        Returns:
            torch.Tensor: loss
        """
        self.optimizer.zero_grad()
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                output = self.model(data, target)
                loss = output.loss
                output = output.logits 
                if isinstance(loss, tuple):
                    loss, loss_info = loss[0], loss[1:]
                self.scaler.scale(loss).backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

        else:
            
            output = self.model(data, target)
            loss = output.loss
            output = output.logits 
            if isinstance(loss, tuple):
                loss, loss_info = loss[0], loss[1:]
            loss.backward()
            self.optimizer.step()
        
        if self.ema:
            try:
                self.ema.update(self.model)
            except RuntimeError:
                self.ema = ModelEMA(self.model)
                self.ema.update(self.model)

        if self._device.type == "cuda":
            torch.cuda.synchronize() # synchronize after each forward and backward pass


        
        self.metrics.update(target, output, model=self.model, loss_info=loss_info)

        return loss
    
    def test_step(
        self, 
        data: typing.Union[np.ndarray, torch.Tensor], 
        target: typing.Union[np.ndarray, torch.Tensor],
        loss_info: dict = {}
        ) -> torch.Tensor:
        """ Perform one validation step

        Args:
            data (typing.Union[np.ndarray, torch.Tensor]): validation data
            target (typing.Union[np.ndarray, torch.Tensor]): validation target
            loss_info (dict, optional): additional loss information. Defaults to {}.

        Returns:
            torch.Tensor: loss
        """
        # output = self.ema.ema(data) if self.ema else self.model(data)
        # loss = self.loss(output, target)
        self.model.eval() 
        with torch.no_grad():
            loss = self.model(data, target).loss 
            
        output = self.model.model.generate(data) 
        # if isinstance(loss, tuple):
        #     loss, loss_info = loss[0], loss[1:]
        self.metrics.update(target, output, model=self.model, loss_info=loss_info)

        # clear GPU memory cache after each validation step
        torch.cuda.empty_cache()

        return loss 
    
    
    
class TrOCRLabelIndexer(LabelIndexer): 
    """Convert label to index by vocab
    
    Attributes:
        vocab (typing.List[str]): List of characters in vocab
    """
    def __init__(
        self, 
        processor
        ) -> None:
        self.processor = processor 

    def __call__(self, data: np.ndarray, label: np.ndarray):
        labels = self.processor.tokenizer(label,
                                          padding="max_length",
                                          max_length= 8 ).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        new_data = self.processor(images=data._image, return_tensors="pt").pixel_values
        new_data = new_data.squeeze(0)  
        return new_data, labels 
    


class CERMetricTrOCR(Metric):
    """A custom PyTorch metric to compute the Character Error Rate (CER).
    
    Args:
        vocabulary: A string of the vocabulary used to encode the labels.
        name: (Optional) string name of the metric instance.

    # TODO: implement everything in Torch to avoid converting to numpy
    """
    def __init__(
        self, 
        processor,
        name: str = "CER"
    ) -> None:
        super(CERMetricTrOCR, self).__init__(name=name)
        self.processor = processor
        self.reset()

    def reset(self):
        """ Reset metric state to initial values"""
        self.cer = 0
        self.counter = 0

    def update(self, output: torch.Tensor, target: torch.Tensor, **kwargs) -> None:
        """ Update metric state with new data

        Args:
            output (torch.Tensor): output of model
            target (torch.Tensor): target of data
        """
        # convert to numpy
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        
        output = np.argmax(output, axis=-1)
        # argmax_preds = np.argmax(output, axis=-1)
        output_texts = self.processor.batch_decode(output, skip_special_tokens=True)
        target[target == -100] = self.processor.tokenizer.pad_token_id
        # print(target)
        target_texts = self.processor.batch_decode(target, skip_special_tokens=True)

        # convert indexes to strings
        # output_texts = ["".join([self.vocabulary[k] for k in group if k < len(self.vocabulary)]) for group in grouped_preds]
        # target_texts = ["".join([self.vocabulary[k] for k in group if k < len(self.vocabulary)]) for group in target]
        # print("output ", output_texts)
        # print("target " , target_texts)
        # print("")
        # print("")
        cer = get_cer(output_texts, target_texts)

        self.cer += cer
        self.counter += 1

    def result(self) -> float:
        """ Return metric value"""
        return self.cer / self.counter