import torch
import transformers
from pe.logging import execution_logger
from fastchat.model.model_adapter import get_conversation_template
from transformers import AutoImageProcessor, AutoModelForImageClassification
from accelerate import init_empty_weights
from pesgd.validators import ModuleValidator


from ..llm import LLM


class HuggingfaceImageModel(LLM):
    """A wrapper for Huggingface LLMs."""

    def __init__(self, model_name_or_path, batch_size=128, dry_run=False, device_map='auto',  gen_with_instruction=False, num_labels=10, ignore_mismatched_sizes=True, **generation_args):
        """Constructor.

        :param model_name_or_path: The model name or path of the Huggingface model. Note that we use the FastChat
        :type model_name_or_path: str
        :param batch_size: The batch size to use for generating the responses, defaults to 128
        :type batch_size: int, optional
        :param dry_run: Whether to enable dry run. When dry run is enabled, the responses are fake and the LLMs are
            not called. Defaults to False
        :type dry_run: bool, optional
        :param \\*\\*generation_args: The generation arguments that will be passed to the OpenAI API
        :type \\*\\*generation_args: str
        """
        self._dry_run = dry_run
        # self._generation_args = generation_args
        # self._gen_with_instruction = gen_with_instruction

        self._model_name_or_path = model_name_or_path
        self._batch_size = batch_size

        self._num_classes = num_labels
        self._ignore_mismatched_sizes = ignore_mismatched_sizes

        self._tokenizer = transformers.AutoImageProcessor.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=True, use_fast=True)
        # if self._tokenizer.pad_token is None:
        #     self._tokenizer.pad_token = self._tokenizer.eos_token
        # self._tokenizer.padding_side = "left"

        self._model = transformers.AutoModelForImageClassification.from_pretrained(
            model_name_or_path, device_map=device_map, torch_dtype=torch.float32, trust_remote_code=True, num_labels=self._num_classes, ignore_mismatched_sizes=self._ignore_mismatched_sizes,
        )
        # # train from scratch
        # config = self._model.config
        # print(f"in <./pe/llm/huggingface/huggingface_image.py> {config=}, {self._model=}")
        # self._model = transformers.AutoModelForImageClassification.from_config(
        #     config, trust_remote_code=True
        # )
        if not ModuleValidator.is_valid(self._model):
            self._model = ModuleValidator.fix(self._model)

        self._model.eval()

    @property
    def generation_arg_map(self):
        """Get the mapping from the generation arguments to arguments for this specific LLM.

        :return: The mapping that maps ``max_completion_tokens`` to ``max_new_tokens``
        :rtype: dict
        """
        # return {"max_completion_tokens": "max_new_tokens"}
        return {}

    def get_responses(self, requests, **generation_args):
        """Get the responses from the LLM.

        :param requests: The requests
        :type requests: list[:py:class:`pe.llm.Request`]
        :param \\*\\*generation_args: The generation arguments. The priority of the generation arguments from the
            highest to the lowerest is in the order of: the arguments set in the requests > the arguments passed to
            this function > and the arguments passed to the constructor
        :type \\*\\*generation_args: str
        :return: The responses
        :rtype: list[str]
        """
        return None
