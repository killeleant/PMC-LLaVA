import json
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
    InstructBlipQFormerConfig,
    InstructBlipQFormerModel,
    InstructBlipProcessor,
)


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        self.use_q = args.use_q

        if self.use_q:
            config_path = "qformer_config.json"
            with open(config_path, "r") as config_file:
                loaded_config = json.load(config_file)
            self.Qformer = InstructBlipQFormerModel(
                InstructBlipQFormerConfig(**loaded_config)
            )
            self.Qformer.load_state_dict(torch.load("qformer_weights.pth"))
            # self.QformerTokenizer = AutoTokenizer.from_pretrained(
            #     "Salesforce/instructblip-flan-t5-xxl"
            # )
            #self.QformerProcessor = InstructBlipProcessor.from_pretrained(
            #    "Salesforce/instructblip-flan-t5-xxl"
            #)

            self.query_tokens = nn.Parameter(torch.load("ROCOquery_tokens.pth"))
            self.projection = nn.Linear(1024, out_features=1408)
            self.post_projection = nn.Linear(
                loaded_config["hidden_size"], out_features=1024
            )

        if not delay_load:
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)


    def load_model(self, device_map=None):
        if self.is_loaded:
            print(
                "{} is already loaded, `load_model` called again, skipping.".format(
                    self.vision_tower_name
                )
            )
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name
        )
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name, device_map=device_map
        )
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    
    def feature_select(self, use_q, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if use_q:
            assert image_features.shape[0] % 10 == 0
            image_features = image_features[0::10]
        else:
            if self.select_feature == "patch":
                image_features = image_features[:, 1:]
            elif self.select_feature == "cls_patch":
                image_features = image_features
            else:
                raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    @torch.no_grad()
    def forward(self, images, qformer_inputids, qf_attention_mask):

        if type(images) is list:
            image_features = []
            query_outputs = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(self.use_q, image_forward_out).to(image.dtype)
                
                image_features.append(image_feature)
                
                if self.use_q:
                    image_feature = self.projection(image_feature)
                    image_attention_mask = torch.ones(image_feature.size()[:-1], dtype=torch.long,
                                                  device=image_feature.device)
                    
                    query_tokens = self.query_tokens.expand(image_feature.shape[0], -1, -1)
                    query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long,
                                                  device=image_feature.device)
                    qformer_attention_mask = torch.cat([query_attention_mask, qf_attention_mask], dim=1)

                    query_output = self.Qformer(
                        input_ids=qformer_inputids,
                        attention_mask=qformer_attention_mask,
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_feature,
                        encoder_attention_mask=image_attention_mask,
                    )
                    

                    query_output = query_output[0][:, : query_tokens.size(1), :]
                    query_output = self.post_projection(query_output)
                    query_outputs.append(query_output)



        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(self.use_q, image_forward_outs).to(images.dtype)
            if self.use_q:

                image_features = self.projection(image_features)
                image_attention_mask = torch.ones(image_features.size()[:-1], dtype=torch.long,
                                                  device=image_features.device)
                
                query_tokens = self.query_tokens.expand(image_features.shape[0], -1, -1)
                query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long,
                                                  device=image_features.device)

                qformer_attention_mask = torch.cat([query_attention_mask, qf_attention_mask], dim=1)

                query_outputs = self.Qformer(
                    input_ids=qformer_inputids,
                    attention_mask=qformer_attention_mask,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_features,
                    encoder_attention_mask=image_attention_mask,
                )


                query_outputs = query_outputs[0][:, : query_tokens.size(1), :]
                query_outputs = self.post_projection(query_outputs)

        return query_outputs if self.use_q else image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, "s2_scales", "336,672,1008")
        self.s2_scales = list(map(int, self.s2_scales.split(",")))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError(
                "Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git"
            )
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, "unfreeze_mm_vision_tower", False):
            self.image_processor.size["shortest_edge"] = self.s2_image_size
            self.image_processor.crop_size["height"] = self.image_processor.crop_size[
                "width"
            ] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(
                "{} is already loaded, `load_model` called again, skipping.".format(
                    self.vision_tower_name
                )
            )
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.vision_tower_name
        )
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name, device_map=device_map
        )
        self.vision_tower.requires_grad_(False)

        self.image_processor.size["shortest_edge"] = self.s2_image_size
        self.image_processor.crop_size["height"] = self.image_processor.crop_size[
            "width"
        ] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
        )
        image_features = self.feature_select(self.use_q, image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(
                    self.forward_feature,
                    image.unsqueeze(0),
                    img_sizes=self.s2_scales,
                    max_split_size=self.s2_split_size,
                )
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(
                self.forward_feature,
                images,
                img_sizes=self.s2_scales,
                max_split_size=self.s2_split_size,
            )

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
