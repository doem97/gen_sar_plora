python ./lora_size.py \
--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
--lora_rank=1 \
--project_name="ORS_LoRA" \
--wandb_note="512x512,stable-diffusion-v1-5,lr=1e-04,100ep,mix_fp16,add keywords for shiprs/dotars" \
--train_data_dir="./lora/data/dosrs_v1/dosrs" \
--dataloader_num_workers="16" \
--resolution="512" \
--center_crop \
--random_flip \
--train_batch_size="24" \
--gradient_accumulation_steps="1" \
--max_train_steps="5540" \
--learning_rate="1e-04" \
--max_grad_norm="1" \
--lr_scheduler="cosine" \
--lr_warmup_steps="531" \
--output_dir="./lora/output/debug/dosrsv2_512_sd15_lr1e-04" \
--cache_dir="./lora/data/cache" \
--report_to="wandb" \
--checkpointing_steps="270" \
--validation_epochs="1" \
--validation_prompt="ors,cargo ship" \
--enable_xformers_memory_efficient_attention \
--seed="42"

python ./lora_size.py \
--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
--lora_rank=2 \
--project_name="ORS_LoRA" \
--wandb_note="512x512,stable-diffusion-v1-5,lr=1e-04,100ep,mix_fp16,add keywords for shiprs/dotars" \
--train_data_dir="./lora/data/dosrs_v1/dosrs" \
--dataloader_num_workers="16" \
--resolution="512" \
--center_crop \
--random_flip \
--train_batch_size="24" \
--gradient_accumulation_steps="1" \
--max_train_steps="5540" \
--learning_rate="1e-04" \
--max_grad_norm="1" \
--lr_scheduler="cosine" \
--lr_warmup_steps="531" \
--output_dir="./lora/output/debug/dosrsv2_512_sd15_lr1e-04" \
--cache_dir="./lora/data/cache" \
--report_to="wandb" \
--checkpointing_steps="270" \
--validation_epochs="1" \
--validation_prompt="ors,cargo ship" \
--enable_xformers_memory_efficient_attention \
--seed="42"

python ./lora_size.py \
--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
--lora_rank=4 \
--project_name="ORS_LoRA" \
--wandb_note="512x512,stable-diffusion-v1-5,lr=1e-04,100ep,mix_fp16,add keywords for shiprs/dotars" \
--train_data_dir="./lora/data/dosrs_v1/dosrs" \
--dataloader_num_workers="16" \
--resolution="512" \
--center_crop \
--random_flip \
--train_batch_size="24" \
--gradient_accumulation_steps="1" \
--max_train_steps="5540" \
--learning_rate="1e-04" \
--max_grad_norm="1" \
--lr_scheduler="cosine" \
--lr_warmup_steps="531" \
--output_dir="./lora/output/debug/dosrsv2_512_sd15_lr1e-04" \
--cache_dir="./lora/data/cache" \
--report_to="wandb" \
--checkpointing_steps="270" \
--validation_epochs="1" \
--validation_prompt="ors,cargo ship" \
--enable_xformers_memory_efficient_attention \
--seed="42"

python ./lora_size.py \
--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
--lora_rank=8 \
--project_name="ORS_LoRA" \
--wandb_note="512x512,stable-diffusion-v1-5,lr=1e-04,100ep,mix_fp16,add keywords for shiprs/dotars" \
--train_data_dir="./lora/data/dosrs_v1/dosrs" \
--dataloader_num_workers="16" \
--resolution="512" \
--center_crop \
--random_flip \
--train_batch_size="24" \
--gradient_accumulation_steps="1" \
--max_train_steps="5540" \
--learning_rate="1e-04" \
--max_grad_norm="1" \
--lr_scheduler="cosine" \
--lr_warmup_steps="531" \
--output_dir="./lora/output/debug/dosrsv2_512_sd15_lr1e-04" \
--cache_dir="./lora/data/cache" \
--report_to="wandb" \
--checkpointing_steps="270" \
--validation_epochs="1" \
--validation_prompt="ors,cargo ship" \
--enable_xformers_memory_efficient_attention \
--seed="42"

python ./lora_size.py \
--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
--lora_rank=16 \
--project_name="ORS_LoRA" \
--wandb_note="512x512,stable-diffusion-v1-5,lr=1e-04,100ep,mix_fp16,add keywords for shiprs/dotars" \
--train_data_dir="./lora/data/dosrs_v1/dosrs" \
--dataloader_num_workers="16" \
--resolution="512" \
--center_crop \
--random_flip \
--train_batch_size="24" \
--gradient_accumulation_steps="1" \
--max_train_steps="5540" \
--learning_rate="1e-04" \
--max_grad_norm="1" \
--lr_scheduler="cosine" \
--lr_warmup_steps="531" \
--output_dir="./lora/output/debug/dosrsv2_512_sd15_lr1e-04" \
--cache_dir="./lora/data/cache" \
--report_to="wandb" \
--checkpointing_steps="270" \
--validation_epochs="1" \
--validation_prompt="ors,cargo ship" \
--enable_xformers_memory_efficient_attention \
--seed="42"

python ./lora_size.py \
--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
--lora_rank=32 \
--project_name="ORS_LoRA" \
--wandb_note="512x512,stable-diffusion-v1-5,lr=1e-04,100ep,mix_fp16,add keywords for shiprs/dotars" \
--train_data_dir="./lora/data/dosrs_v1/dosrs" \
--dataloader_num_workers="16" \
--resolution="512" \
--center_crop \
--random_flip \
--train_batch_size="24" \
--gradient_accumulation_steps="1" \
--max_train_steps="5540" \
--learning_rate="1e-04" \
--max_grad_norm="1" \
--lr_scheduler="cosine" \
--lr_warmup_steps="531" \
--output_dir="./lora/output/debug/dosrsv2_512_sd15_lr1e-04" \
--cache_dir="./lora/data/cache" \
--report_to="wandb" \
--checkpointing_steps="270" \
--validation_epochs="1" \
--validation_prompt="ors,cargo ship" \
--enable_xformers_memory_efficient_attention \
--seed="42"

python ./lora_size.py \
--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
--lora_rank=64 \
--project_name="ORS_LoRA" \
--wandb_note="512x512,stable-diffusion-v1-5,lr=1e-04,100ep,mix_fp16,add keywords for shiprs/dotars" \
--train_data_dir="./lora/data/dosrs_v1/dosrs" \
--dataloader_num_workers="16" \
--resolution="512" \
--center_crop \
--random_flip \
--train_batch_size="24" \
--gradient_accumulation_steps="1" \
--max_train_steps="5540" \
--learning_rate="1e-04" \
--max_grad_norm="1" \
--lr_scheduler="cosine" \
--lr_warmup_steps="531" \
--output_dir="./lora/output/debug/dosrsv2_512_sd15_lr1e-04" \
--cache_dir="./lora/data/cache" \
--report_to="wandb" \
--checkpointing_steps="270" \
--validation_epochs="1" \
--validation_prompt="ors,cargo ship" \
--enable_xformers_memory_efficient_attention \
--seed="42"

python ./lora_size.py \
--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
--lora_rank=128 \
--project_name="ORS_LoRA" \
--wandb_note="512x512,stable-diffusion-v1-5,lr=1e-04,100ep,mix_fp16,add keywords for shiprs/dotars" \
--train_data_dir="./lora/data/dosrs_v1/dosrs" \
--dataloader_num_workers="16" \
--resolution="512" \
--center_crop \
--random_flip \
--train_batch_size="24" \
--gradient_accumulation_steps="1" \
--max_train_steps="5540" \
--learning_rate="1e-04" \
--max_grad_norm="1" \
--lr_scheduler="cosine" \
--lr_warmup_steps="531" \
--output_dir="./lora/output/debug/dosrsv2_512_sd15_lr1e-04" \
--cache_dir="./lora/data/cache" \
--report_to="wandb" \
--checkpointing_steps="270" \
--validation_epochs="1" \
--validation_prompt="ors,cargo ship" \
--enable_xformers_memory_efficient_attention \
--seed="42"
