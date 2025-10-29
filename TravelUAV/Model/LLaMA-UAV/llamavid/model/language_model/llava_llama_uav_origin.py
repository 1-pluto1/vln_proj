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
 
        
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output):
        output_reshaped = output.view(output.size(0), 3, 7)
        norms = torch.norm(output_reshaped, dim=1)
        loss = torch.nn.functional.l1_loss(norms, torch.ones_like(norms))
        return loss

class CosineDirectionLoss(nn.Module):
    def __init__(self):
        super(CosineDirectionLoss, self).__init__()
    
    def forward(self, vec1, vec2):
        cosine_sim = F.cosine_similarity(vec1, vec2, dim=-1)
        loss = 1 - cosine_sim
        return loss.mean()
    
class ScaleL1Loss(nn.Module):
    def __init__(self, scale=[1, 1, 1]):
        super(ScaleL1Loss, self).__init__()
        self.base_scale = torch.tensor(scale)
        self.gaussian_scale = self.create_gaussian_scale()

    def create_gaussian_scale(self):
        center = 0 
        sigma = 1.5
        x = torch.arange(7, dtype=torch.bfloat16)
        gaussian_scale = torch.exp(-0.5 * ((x - center) ** 2) / sigma ** 2)
        return gaussian_scale

    def is_updown(self, target):
        batch_size = target.size(0)
        base_scale = torch.ones(batch_size, 3).to(dtype=target.dtype, device=target.device)
        for i in range(batch_size):
            distance = torch.abs(target[i, 0, 2] - target[i, 6, 2])
            if distance > 5:
                base_scale[i] = torch.tensor([1, 1, 5]).to(dtype=target.dtype, device=target.device)
            else:
                base_scale[i] = torch.tensor([1, 1, 1]).to(dtype=target.dtype, device=target.device)
        base_scale = base_scale.view(batch_size, 1, 3).expand_as(target)
        return base_scale
    
    def is_turn(self, target):
        batch_size = target.size(0)
        base_scale = torch.ones(batch_size, 3).to(dtype=target.dtype, device=target.device)
        for i in range(batch_size):
            vec_1 = target[i, 0, :]
            vec_2 = target[i, 1, :] - target[i, 0, :]
            try:
                cos = vec_1.dot(vec_2) / (torch.norm(vec_1) * torch.norm(vec_2) + 1e-6)
                if cos < 0.85:
                    base_scale[i] = torch.tensor([5, 5, 1]).to(dtype=target.dtype, device=target.device)
                else:
                    base_scale[i] = torch.tensor([1, 1, 1]).to(dtype=target.dtype, device=target.device)
            except:
                base_scale[i] = torch.tensor([1, 1, 1]).to(dtype=target.dtype, device=target.device)
        base_scale = base_scale.view(batch_size, 1, 3).expand_as(target)
        return base_scale
    
    def forward(self, output, target, base_scale=None):
        # import ipdb; ipdb.set_trace()
        output = output.view(output.size(0), 7, 3)
        target = target.view(target.size(0), 7, 3)
        base_scale = self.base_scale.view(1, 1, 3).expand_as(output).to(dtype=output.dtype, device=output.device)
        base_scale_updown = self.is_updown(target)
        base_scale_turn = self.is_turn(target)
        base_scale = base_scale * base_scale_updown * base_scale_turn
        gaussian_scale = self.gaussian_scale.view(1, 7, 1).expand_as(output).to(dtype=output.dtype, device=output.device)
        scale = base_scale * gaussian_scale
        loss = torch.abs(output - target)
        x_loss = loss[:,:,0].mean()
        y_loss = loss[:,:,1].mean()
        z_loss = loss[:,:,2].mean()
        
        only_predict_one_point = True
        if only_predict_one_point:
            mask = torch.zeros(output.size(0), 7, 3).to(dtype=output.dtype, device=output.device)
            mask[:, 0, :] = 1
            scale = scale * mask
            
        weighted_loss = loss * scale
        ori_loss = loss.mean()
        weighted_loss = weighted_loss.mean()
        return {'loss': weighted_loss,
                'ori_loss': ori_loss,
                'x_loss': x_loss,
                'y_loss': y_loss,
                'z_loss': z_loss}
    

class LlavaLlamaAttForCausalLM(LlamaUAVForCausalLM, LLaMAVIDMetaForCausalLM):
    config_class = LlavaConfig
    _keep_in_fp32_modules = ['waypoints_predictor']
    def __init__(self, config):
        super(LlamaUAVForCausalLM, self).__init__(config)
        self.model = LlavaAttLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.waypoint_emb = nn.Embedding(1, config.hidden_size)
        self.waypoints_fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 64),
        )
        self.waypoints_predictor = nn.GRUCell(input_size=3, hidden_size=64)
        self.waypoints_output = nn.Linear(64, 4)
        
        self.is_help_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 2)
        )
        
        # self.end_predictor = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.hidden_size // 2),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_size // 2, 2)
        # )
        
        self.history_predictor = nn.Sequential(
            nn.Linear(4096, 4096 // 2),
            nn.ReLU(),
            nn.Linear(4096 // 2, 12)
        )
        
        self.history_preprocessor = nn.Sequential(
            nn.Linear(3, 4096 // 2),
            nn.ReLU(),
            nn.Linear(4096 // 2, 4096),
        )
        
        self.waypoints_loss_func = torch.nn.L1Loss()
        self.angle_loss_func = CosineDirectionLoss()
        self.scale_waypoints_loss_func = ScaleL1Loss()
        self.waypoint_loss_scale = 1.0
        
        self.end_loss_func = torch.nn.CrossEntropyLoss()
        self.end_loss_scale = 1.0
        
        self.is_help_loss_func = torch.nn.CrossEntropyLoss()
        self.is_help_loss_scale = 1.0
        
        self.history_loss_func = torch.nn.L1Loss()
        self.history_loss_scale = 1.0
        
        self.custom_loss_func = CustomLoss()
        self.custom_loss_scale = 10.0
        
        self.special_token_dict = None

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_special_token_id(self, special_token_dict):
        self.special_token_dict = special_token_dict
        
    def get_model(self):
        return self.model
    
    def forward_waypoint(self, hidden_states):
        output_wp = [] #waypoints
        output_delta = [] #deltas
        bs, hidden_size = hidden_states.size()
        waypoints_feature = self.waypoints_fc(hidden_states.reshape(-1, hidden_size))
        
        predicted_waypoints = self.waypoints_output(waypoints_feature)
        predicted_deltas = predicted_waypoints
        # waypoints_feature = torch.cat([waypoints_feature, historys], dim=1)
        # x = torch.zeros(size=(bs, 3), dtype=hidden_states.dtype).to(hidden_states.device)
        # for _ in range(7):
        #     x_in = x
        #     with torch.autocast(device_type='cuda', dtype=torch.float32):
        #         waypoints_feature = self.waypoints_predictor(x_in, waypoints_feature)
        #     dx = self.waypoints_output(waypoints_feature.to(hidden_states.dtype))
        #     x = dx + x
        #     output_wp.append(x)
        #     output_delta.append(dx)
        # predicted_waypoints = torch.cat(output_wp, dim=1)
        # predicted_deltas = torch.cat(output_delta, dim=1)
        # predicted_waypoints = predicted_waypoints.view(bs, 21)
        # predicted_deltas = predicted_deltas.view(bs, 21)
        
        return predicted_waypoints, predicted_deltas
    
    def forward_end(self, hidden_states):
        # end_logits = self.end_predictor(hidden_states)
        # return end_logits
        return None
    
    def forward_is_help(self, hidden_states):
        # MLP for positive
        is_help_logits = self.is_help_predictor(hidden_states)
        
        return is_help_logits
    
    def forward_history(self, hidden_states):
        # MLP for positive
        history_logits = -self.history_predictor(hidden_states)
        return history_logits

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
        ends: Optional[torch.FloatTensor] = None,
        is_helps: Optional[torch.FloatTensor] = None,
        historys: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        return_waypoints: Optional[bool] = False,
        use_custom_loss: Optional[bool] = False,
        use_scale_waypoint_loss: Optional[bool] = False,
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
        
        v3= True
        for idx in range(len(historys)):
            history = historys[idx]
            orientation = orientations[idx]
            if v3:
                info = history.view(-1, 3)
            else:
                info = torch.cat([orientation, history], dim=0).view(-1, 3)
            history_embed = self.history_preprocessor(info)
            history_embeds.append(history_embed)
            
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, prompts=prompts, historys=history_embeds, special_token_dict=self.special_token_dict)
        # import ipdb;ipdb.set_trace()
        inputs_embeds = inputs_embeds.to(dtype=self.waypoint_emb.weight.dtype)
        inputs_embeds[labels == WAYPOINT_LABEL_TOKEN] = self.waypoint_emb.weight
        
        torch.cuda.empty_cache()
        
        # import ipdb; ipdb.set_trace()
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(#llava
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
        # import ipdb; ipdb.set_trace()
        torch.cuda.empty_cache()
        
        waypoints_feat = hidden_states[labels == WAYPOINT_LABEL_TOKEN]     
        predicted_waypoints, predicted_deltas = self.forward_waypoint(waypoints_feat)
        # predicted_end = self.forward_end(waypoints_feat)
        predicted_is_help = self.forward_is_help(waypoints_feat)
        
        if waypoints is None and return_waypoints:
            return predicted_waypoints, predicted_is_help
        
        loss = None
        loss_dict = None
        
        assert len(torch.where(labels == WAYPOINT_LABEL_TOKEN)[0]) == waypoints.shape[0]
        
        use_scale_waypoint_loss = False
        use_angle_and_norm_loss = True
        use_custom_loss = False
        
        if waypoints is not None:
            # import ipdb; ipdb.set_trace()
            
            if use_scale_waypoint_loss:
                loss_dict = self.scale_waypoints_loss_func(predicted_waypoints, waypoints)
                print('predict:\n', predicted_waypoints[:,:3])
                print('waypoints:\n', waypoints[:,:3])
                waypoint_loss =  loss_dict['loss']
                waypoint_loss = self.waypoint_loss_scale * waypoint_loss
            elif use_angle_and_norm_loss:
                print('predict:\n', predicted_waypoints)
                print('waypoints:\n', waypoints)
                waypoint_loss = self.waypoint_loss_scale * self.waypoints_loss_func(predicted_waypoints[:, 3], waypoints[:, 3])
                angle_loss = self.waypoint_loss_scale * self.angle_loss_func(predicted_waypoints[:, :3], waypoints[:, :3])
                waypoint_loss = waypoint_loss + angle_loss
            else:
                waypoint_loss = self.waypoint_loss_scale * self.waypoints_loss_func(predicted_waypoints, waypoints) 

            # end_loss = self.end_loss_scale * self.end_loss_func(predicted_end, ends)
            
            if use_custom_loss:
                custom_loss = self.custom_loss_scale * self.custom_loss_func(predicted_deltas)
                loss = waypoint_loss + custom_loss
                print(f"Loss info --- \t waypoint_loss : {waypoint_loss} \t custom_loss : {custom_loss}.")
            elif is_helps is not None:
                # import ipdb; ipdb.set_trace()
                print('is_help:\n', predicted_is_help)
                is_help_loss = self.is_help_loss_scale * self.is_help_loss_func(predicted_is_help, is_helps)
                
                loss_dict = {'loss': waypoint_loss,
                            'ori_loss': waypoint_loss,
                            'x_loss': waypoint_loss,
                            'y_loss': waypoint_loss,
                            'z_loss': waypoint_loss}
                
                loss_dict['help_loss'] = is_help_loss
                
                # 先去掉help loss
                loss = waypoint_loss
                print(f"Loss info --- \t waypoint_loss : {waypoint_loss} \t is_help_loss : {is_help_loss} \t total_loss: {loss}")
            else:
                loss = waypoint_loss
                print(f"Loss info --- \t weighted_loss : {waypoint_loss}.\t ori_loss : {loss_dict['ori_loss']}.")
            # loss = self.waypoint_loss_scale * self.waypoints_loss_func(predicted_waypoints, waypoints) + \
            #     self.end_loss_scale * self.end_loss_func(predicted_end, ends) + \
            #     self.history_loss_scale * self.history_loss_func(predicted_history, historys)
            
        # logits = self.lm_head(hidden_states)
        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     shift_logits = shift_logits.view(-1, self.config.vocab_size)
        #     shift_labels = shift_labels.view(-1)
        #     # Enable model/pipeline parallelism
        #     shift_labels = shift_labels.to(shift_logits.device)
        #     loss = loss_fct(shift_logits, shift_labels)
        
        import ipdb;ipdb.set_trace()
        if return_waypoints:
            return loss, predicted_waypoints
        
        if not return_dict:
            output = (waypoints_feat,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        # import ipdb; ipdb.set_trace()
        return CausalLMOutputWithPastUAVMulLoss(
            ori_loss=loss_dict['ori_loss'],
            loss=loss,
            x_loss=loss_dict['x_loss'],
            y_loss=loss_dict['y_loss'],
            z_loss=loss_dict['z_loss'],
            help_loss=loss_dict['help_loss'],
            # past_key_values=outputs.past_key_values,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
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
