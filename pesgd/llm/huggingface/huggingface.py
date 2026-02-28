import torch
import transformers
from pe.logging import execution_logger
from fastchat.model.model_adapter import get_conversation_template
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

from ..llm import LLM
# from ..fine_tune.instruction_addition import chat_template_tokenize_example
from .model_name_to_path import MODEL_NAME_TO_PATH


class HuggingfaceLLM(LLM):
    """A wrapper for Huggingface LLMs."""

    def __init__(self, model_name_or_path, batch_size=128, dry_run=False, device_map='auto',  gen_with_instruction=False, use_local_model=False, **generation_args):
        """Constructor.

        :param model_name_or_path: The model name or path of the Huggingface model. Note that we use the FastChat
            library (https://github.com/lm-sys/FastChat) to manage the conversation template. If the conversation
            template of your desired model is not available in FastChat, please register the conversation template in
            the FastChat library. See the following link for an example:
            https://github.com/microsoft/DPSDA/blob/main/pe/llm/huggingface/register_fastchat/gpt2.py
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
        self._generation_args = generation_args
        self._gen_with_instruction = gen_with_instruction

        self._model_name_or_path = model_name_or_path
        print(f"{self._model_name_or_path=}, {use_local_model=}")
        if use_local_model:
            model_name_or_path = MODEL_NAME_TO_PATH[model_name_or_path]
        self._batch_size = batch_size

        if use_local_model and 'gpt2' in model_name_or_path:
            print(f"Using local gpt2 model from {model_name_or_path}")
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=True, model_max_length=1024)
        else:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        # if not 'gpt2' in self._model_name_or_path:
        #     self._tokenizer.padding_side = "right" # use right padding for training.
        # else:
        #     self._tokenizer.padding_side = "left"
        print(f"{self._tokenizer.model_max_length=}")
        if self._tokenizer.chat_template == None:
            if not 'gpt2' in self._model_name_or_path:
                print(f"here, {self._model_name_or_path=}")
                self._tokenizer.chat_template = """{% for message in messages %}
                    {% if message['role'] == 'system' %}System: {{ message['content'] }}
                    {% elif message['role'] == 'user' %}User: {{ message['content'] }}
                    {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}
                    {% endif %}
                    {% endfor %}"""
            else:
                self._tokenizer.chat_template = """{% for message in messages %}
                    {% if message['role'] == 'system' %} {{ message['content'] }}
                    {% elif message['role'] == 'user' %} Instruction: {{ message['content'] }}
                    {% elif message['role'] == 'assistant' %} Response: {{ message['content'] }}
                    {% endif %}
                    {% endfor %}"""
                # self._tokenizer.add_special_tokens({
                #     "pad_token": "<PAD>",
                #     "bos_token": "<BOS>",
                #     "eos_token": "<EOS>"
                # })
        else:
            print(f"Using the conversation template in the tokenizer")
            # print(f"Using the conversation template in the tokenizer, {self._tokenizer.chat_template}")

        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map=device_map, torch_dtype=torch.float16, trust_remote_code=True,
        )
        if 'gpt2' in self._model_name_or_path:
            self._model.resize_token_embeddings(len(self._tokenizer))
        # with init_empty_weights():
        #     # config = transformers.AutoConfig.from_pretrained(model_name_or_path)
        #     # self._model = transformers.AutoModelForCausalLM.from_config(config)
        #     self._model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
        
        if self._model.config.pad_token_id is None:
            print(f"Setting pad_token_id to eos_token_id for model {model_name_or_path} since pad_token_id is not set.")
            self._model.config.pad_token_id = self._model.config.eos_token_id
        self._model.eval()

        self._conv_template = self._get_conv_template()
        if self._conv_template.name == "one_shot":
            execution_logger.warning(
                "The conversation template is the default one_shot. Likely the conversation template is not set "
                "correctly. Please check if the installed fastchat library is the latest version on GitHub, or if the "
                "conversation template is registered. See "
                "https://microsoft.github.io/DPSDA/api/pe.llm.html#pe.llm.HuggingfaceLLM"
            )
        self._stop_str = self._conv_template.stop_str
        self._stop_token_ids = self._conv_template.stop_token_ids or []
        self._stop_token_ids.append(self._tokenizer.eos_token_id)

    @property
    def generation_arg_map(self):
        """Get the mapping from the generation arguments to arguments for this specific LLM.

        :return: The mapping that maps ``max_completion_tokens`` to ``max_new_tokens``
        :rtype: dict
        """
        return {"max_completion_tokens": "max_new_tokens"}

    def _get_conv_template(self):
        """Get the conversation template.

        :return: The empty conversation template for this model from FastChat
        :rtype: :py:class:`fastchat.conversation.Conversation`
        """
        # if self._tokenizer.chat_template is not None:
        #     execution_logger.info("HuggingfaceLLM: using the conversation template in the tokenizer")
        #     template = self._tokenizer.get_chat_template()
        # else:
        #     execution_logger.info("HuggingfaceLLM: using the conversation template in the fastchat library")
        template = get_conversation_template(self._model_name_or_path)
        template.messages = []
        template.system_message = ""
        return template

    def _get_prompt(self, messages):
        """Get the prompt from the messages.

        :param messages: The messages
        :type messages: list[dict]
        :raises ValueError: If the role is invalid
        :return: The prompt
        :rtype: str
        """
        template = self._conv_template.copy()
        for message in messages:
            if message["role"] == "system":
                template.set_system_message(message["content"])
            elif message["role"] == "user":
                template.append_message(role=template.roles[0], message=message["content"])
            elif message["role"] == "assistant":
                template.append_message(role=template.roles[1], message=message["content"])
            else:
                raise ValueError(f"Invalid role: {message['role']}")
        template.append_message(role=template.roles[1], message=None)
        return template.get_prompt()

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
        execution_logger.info("HuggingfaceLLM: producing prompts")
        prompt_list = []
        generation_args_list = []
        for request in requests:
            prompt_list.append(self._get_prompt(request.messages))
            generation_args_list.append(
                self.get_generation_args(self._generation_args, generation_args, request.generation_args)
            )
        execution_logger.info("HuggingfaceLLM: getting responses")
        responses = [None] * len(requests)
        # Group requests according to generation_args
        generation_args_fronzen_set_list = [
            frozenset(generation_args.items()) for generation_args in generation_args_list
        ]
        generation_args_set = list(set(generation_args_fronzen_set_list))
        generation_args_to_set_index = {g: i for i, g in enumerate(generation_args_set)}
        grouped_request_indices = [[] for i in range(len(generation_args_set))]
        for i, generation_args in enumerate(generation_args_fronzen_set_list):
            grouped_request_indices[generation_args_to_set_index[generation_args]].append(i)
        for group in grouped_request_indices:
            sub_prompt_list = [prompt_list[j] for j in group]
            sub_response_list = self._get_responses(sub_prompt_list, generation_args_list[group[0]])
            for i, j in enumerate(group):
                responses[j] = sub_response_list[i]
        assert None not in responses
        return responses

    @torch.no_grad
    def _get_responses(self, prompt_list, generation_args):
        """Get the responses from the LLM.

        :param prompt_list: The prompts
        :type prompt_list: list[str]
        :param generation_args: The generation arguments
        :type generation_args: dict
        :return: The responses
        :rtype: list[str]
        """
        # print(f"[debugging] in <huggingface.py> _get_repsonse for generation {prompt_list[0]=}")
        # print(f"[debugging] in <huggingface.py> _get_repsonse for generation {prompt_list[-1]=}")
        if self._dry_run:
            responses = [f"Dry run enabled. The request is {prompt}" for prompt in prompt_list]
        else:
            inputs = self._tokenizer(
                prompt_list, return_tensors="pt", padding=True, padding_side="left", # generation use left padding.
            )

            input_ids = inputs.input_ids.to(self._model.device)
            attention_masks = inputs.attention_mask.to(self._model.device) 
            responses = []
            # print(f"{self._tokenizer.convert_tokens_to_ids('<|eot_id|>')=}")
            for i in range(0, len(input_ids), self._batch_size):
                batch_input_ids = input_ids[i : i + self._batch_size]
                batch_attention_masks = attention_masks[i : i + self._batch_size] if attention_masks is not None else None
                batch_responses = self._model.generate(
                    batch_input_ids,
                    attention_mask=batch_attention_masks,
                    stop_strings=self._stop_str,
                    # eos_token_id=[self._stop_token_ids, self._tokenizer.convert_tokens_to_ids("<|eot_id|>"), self._tokenizer.convert_tokens_to_ids("<|im_end|>")] if '<|eot_id|>' in self._tokenizer.get_vocab() else self._stop_token_ids,
                    eos_token_id=self._stop_token_ids,
                    do_sample=True,
                    tokenizer=self._tokenizer,
                    **generation_args,
                )
                batch_responses = self._tokenizer.batch_decode(
                    batch_responses[:, input_ids.shape[1] :],
                    clean_up_tokenization_spaces=True,
                    skip_special_tokens=True,
                )
                responses.extend(batch_responses)
        return responses
