rsync -avz --include='*/' --exclude='*.safetensors' --exclude='*.pt' [user]@superpod.ust.hk:/home/[user]/[repo_path]/outputs/ ./outputs --progress
# rsync -avz --include='*/' --exclude='*.safetensors' --exclude='*.pt' wjcui@superpod.ust.hk:/home/wjcui/5103/LoRA-Ablation/outputs/ ./outputs --progress
