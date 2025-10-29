#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig

from llamavid.model.llamavid_arch import LLaMAVIDMetaModel, LLaMAVIDMetaForCausalLM
from llamavid.model.language_model.llama_uav import LlamaUAVModel, LlamaUAVForCausalLM, CausalLMOutputWithPastUAV, CausalLMOutputWithPastUAVMulLoss

from llamavid.constants import WAYPOINT_LABEL_TOKEN

class LlavaConfig(LlamaConfig):
    model_type = "llava"

class LlavaAttLlamaModel(LLaMAVIDMetaModel, LlamaUAVModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaAttLlamaModel, self).__init__(config)
 
        
class CosineDirectionLoss(nn.Module):
    def __init__(self):
        super(CosineDirectionLoss, self).__init__()
    
    def forward(self, vec1, vec2):
        cosine_sim = F.cosine_similarity(vec1, vec2, dim=-1)
        loss = 1 - cosine_sim
        return loss.mean()
    

class LlavaLlamaAttForCausalLM(LlamaUAVForCausalLM, LLaMAVIDMetaForCausalLM):
    config_class = LlavaConfig
    def __init__(self, config, **model_args):
        super(LlamaUAVForCausalLM, self).__init__(config)
        self.model = LlavaAttLlamaModel(config)
        self.use_angle_and_norm_loss = model_args.get('use_angle_and_norm_loss', True)
        # self.
        # TODO: set LLaMAVIDMetaForCausalLM config
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.waypoint_emb = nn.Embedding(1, config.hidden_size)
        self.waypoints_fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 64),
        )
        self.waypoints_output = nn.Linear(64, 4)
        
        self.history_preprocessor = nn.Sequential(
            nn.Linear(3, 4096 // 2),
            nn.ReLU(),
            nn.Linear(4096 // 2, 4096),
        )
        
        self.waypoints_loss_func = torch.nn.L1Loss()
        self.angle_loss_func = CosineDirectionLoss()
        self.waypoint_loss_scale = 1.0
        self.special_token_dict = None

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_special_token_id(self, special_token_dict):
        self.special_token_dict = special_token_dict
        
    def get_model(self):
        return self.model
    
    def forward_waypoint(self, hidden_states):
        bs, hidden_size = hidden_states.size()
        waypoints_feature = self.waypoints_fc(hidden_states.reshape(-1, hidden_size))
        
        predicted_waypoints = self.waypoints_output(waypoints_feature)
        return predicted_waypoints

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        prompts: Optional[List[str]] = None,
        waypoints: Optional[torch.FloatTensor] = None,
        orientations: Optional[torch.FloatTensor] = None,
        historys: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        return_waypoints: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPastUAV]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not self.training:
            if images[0].device != self.device:
                if type(images) is not list:
                    images = images.to(device=self.device)
                else:
                    images = [image.to(device=self.device) for image in images]
            if input_ids.device != self.device:
                input_ids = input_ids.to(device=self.device)
            if attention_mask.device != self.device:
                attention_mask = attention_mask.to(device=self.device)
            if labels.device != self.device:
                labels = labels.to(device=self.device)
                
        # import ipdb; ipdb.set_trace()
        if type(images) is not list:
            images = images.to(dtype=self.dtype)
        else:
            images = [image.to(dtype=self.dtype) for image in images]
        
        history_embeds = []
        
        for idx in range(len(historys)):
            history = historys[idx]
            info = history.view(-1, 3)
            history_embed = self.history_preprocessor(info)
            history_embeds.append(history_embed)
            
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, prompts=prompts, historys=history_embeds, special_token_dict=self.special_token_dict)
        inputs_embeds = inputs_embeds.to(dtype=self.waypoint_emb.weight.dtype)
        inputs_embeds[labels == WAYPOINT_LABEL_TOKEN] = self.waypoint_emb.weight
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = outputs[0]
        waypoints_feat = hidden_states[labels == WAYPOINT_LABEL_TOKEN]     
        predicted_waypoints = self.forward_waypoint(waypoints_feat)
        
        if waypoints is None and return_waypoints:
            return predicted_waypoints
        
        loss = None
        
        assert len(torch.where(labels == WAYPOINT_LABEL_TOKEN)[0]) == waypoints.shape[0]
        if waypoints is not None:
            if self.use_angle_and_norm_loss:
                waypoint_loss = self.waypoint_loss_scale * self.waypoints_loss_func(predicted_waypoints[:, 3], waypoints[:, 3])
                angle_loss = self.waypoint_loss_scale * self.angle_loss_func(predicted_waypoints[:, :3], waypoints[:, :3])
                loss = waypoint_loss + angle_loss
            else:
                loss = self.waypoint_loss_scale * self.waypoints_loss_func(predicted_waypoints, waypoints) 
        
        if return_waypoints:
            return loss, predicted_waypoints
        
        if not return_dict:
            output = (waypoints_feat,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPastUAVMulLoss(
            loss=loss,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaAttForCausalLM)
