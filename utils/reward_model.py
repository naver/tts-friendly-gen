from abc import ABC, abstractmethod
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List
from utils.common import get_respond, get_feature_scorer
from utils.prompts import *
import pickle
import numpy as np

class AbstractRewardModel(ABC):
    @abstractmethod
    def score(self, situation: str, action: str) -> float:
        """
        Computes a reward score given a situation and an action.
        
        :param situation: A description of the current situation.
        :param action: A description of the action taken.
        :return: A float representing the reward score.
        """
        pass

class RewardModel(AbstractRewardModel):
    def __init__(self, model_name_or_path: str, data_type_upper: str, use_template: bool):
        """
        Initializes the reward model by loading a pretrained checkpoint.
        
        :param model_name_or_path: The name or path of the pretrained reward model to load.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=1).to("cuda")
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        self.data_type_upper = data_type_upper
        self.use_template = use_template

    def score(self, situation: str, action: str) -> float:
        """
        Computes a reward score using a pretrained reward model.
        
        :param situation: A description of the current situation.
        :param action: A description of the action taken.
        :return: A float representing the reward score.
        """

        sys_text = eval(f"{self.data_type_upper}_GEN_SYS_TEMPLATE")
        user_text = eval(f"{self.data_type_upper}_GEN_ZEROSHOT_USER_TEMPLATE").format(situation=situation)
        user_text += " " + action # Append the action to the prompt containing the situation
            
        if self.use_template:
            messages = [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": user_text}
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False
            ).to("cuda")
        else:
            messages = f"{sys_text}\n\n{user_text} {action}"
            inputs = self.tokenizer(
                messages,
                return_tensors="pt",
                padding=True
            ).to("cuda")

        with torch.no_grad():
            outputs = self.model(inputs["input_ids"])
        score = outputs.logits.squeeze().item()

        return float(score)

class FeatureAwareRewardModel(AbstractRewardModel):
    def __init__(self, rm_path: str, dtype: torch.dtype):
        """
        Initializes the feature-based reward model as a linear combination of feature functions.
        
        :param rm_path: The path to the file containing the feature-based reward model parameters.
        :param model_name: The name of the pretrained LLM to use for the feature functions.
        :param mode: Mode for response generation ('gen', 'max' or 'mean').
        :param data_type: Type of data formatting to be used for feature extraction.
        """
        # Load the feature-based reward model parameters
        with open(rm_path, "rb") as pickle_file:
            rm_params = pickle.load(pickle_file)
        data_type = rm_params['data_type']
        model_name = rm_params['model_name']
        mode = rm_params['mode']
        self.feature_names = rm_params['feature_names']
        self.features = rm_params['features']
        self.features_mean = rm_params['features_mean']['train']
        self.features_std = rm_params['features_std']['train']
        self.coefficients = rm_params['coefficients']
        print({f: c for f, c in zip(self.feature_names, self.coefficients)})

        # Initialize the feature function generator
        respond = get_respond(model_name, "huggingface", mode, 0, dtype) # Seed used only for OpenAI models
        self.feature_scorer = get_feature_scorer(respond, data_type)

    def annotate_features(self, situation: str, action: str) -> float:
        """
        Uses the feature function to compute the feature values for a given situation and action.
        
        :param situation: A description of the situation.
        :param action: A description of the action taken.
        :return: A numpy array representing the feature values.
        """
        # Score the features for this (situation, action) pair
        feature_score_dict = self.feature_scorer(situation, action, self.features)
        feature_values = np.array([feature_score_dict[feature_name] for feature_name in self.feature_names])

        # Whiten the reward using the mean/std obtained during the training of the reward model
        feature_values = (feature_values - self.features_mean) / self.features_std

        return feature_values

    def score(self, situation: str, action: str) -> float:
        """
        Computes the reward score as a linear combination of feature values extracted from the feature function.
        
        :param situation: A description of the situation.
        :param action: A description of the action taken.
        :return: A float representing the reward score.
        """
        # Compute the feature values
        feature_values = self.annotate_features(situation, action)

        # Compute the linear combination of the feature values, weighed by the coefficients
        dot_prod = (self.coefficients * feature_values).sum()

        return dot_prod