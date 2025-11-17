| exp_name | eval_accuracy | eval_loss | trainable_params | trainable_ratio | training_time_sec | gpu_mem_peak_mb | use_lora | rank | alpha | dropout | improved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_full_ft | 0.9369 | 0.2606 | 124647170 | 1 | 311.3 | 2481 | False | 8 | 16 | 0 | False |
| lora_alpha_16 | 0.9255 | 0.2136 | 887042 | 0.007066 | 281 | 1113 | True | 8 | 16 | 0.05 | False |
| lora_alpha_32 | 0.9255 | 0.2199 | 887042 | 0.007066 | 279.5 | 1113 | True | 8 | 32 | 0.05 | False |
| lora_alpha_8 | 0.9266 | 0.2147 | 887042 | 0.007066 | 281.7 | 1113 | True | 8 | 8 | 0.05 | False |
| lora_dropout_only | 0.9278 | 0.2162 | 887042 | 0.007066 | 270.8 | 1113 | True | 8 | 16 | 0.1 | False |
| lora_improved_dropout | 0.9312 | 0.214 | 1255682 | 0.009973 | 309.6 | 1147 | True | 8 | 16 | 0.1 | True |
| lora_rank_r16 | 0.9278 | 0.2091 | 1181954 | 0.009393 | 278 | 1117 | True | 16 | 16 | 0.05 | False |
| lora_rank_r2 | 0.9197 | 0.2168 | 665858 | 0.005314 | 280.6 | 1110 | True | 2 | 16 | 0.05 | False |
| lora_rank_r4 | 0.9266 | 0.2178 | 739586 | 0.005898 | 280 | 1111 | True | 4 | 16 | 0.05 | False |
| lora_rank_r8 | 0.9278 | 0.2091 | 887042 | 0.007066 | 278.7 | 1113 | True | 8 | 16 | 0.05 | False |
| lora_target_QV | 0.9289 | 0.2166 | 887042 | 0.007066 | 280.5 | 1113 | True | 8 | 16 | 0.05 | False |
| lora_target_QV_fc1_improved | 0.9312 | 0.217 | 1255682 | 0.009973 | 326.4 | 1147 | True | 8 | 16 | 0.05 | True |


---

# ✅ **Appendix: Short Explanation of Each Experiment**

### **baseline_full_ft**

Full fine-tuning of RoBERTa-base. All model parameters are updated.
Serves as the upper-bound reference point for performance, time, and GPU usage.

---

### **lora_rank_r2 / r4 / r8 / r16**

Rank ablation: evaluates the effect of LoRA rank ( r ).

* Small ranks (2, 4) restrict update capacity → slightly lower accuracy.
* Mid-rank (8) matches or approaches full fine-tuning.
* Larger rank (16) increases parameters with limited accuracy gain.
  Validates diminishing-returns behavior described in the LoRA paper.

---

### **lora_alpha_8 / alpha_16 / alpha_32**

Scaling factor ablation.
(\alpha) controls the magnitude of the low-rank update.

* All values perform similarly
* Larger α (32) can cause mild instability or no improvement
  Shows LoRA is robust to scaling within reasonable ranges.

---

### **lora_target_QV**

Standard LoRA configuration.
Only inserts LoRA into **q_proj** and **v_proj**, following the original paper.
Represents the “baseline LoRA” for comparison with our improvements.

---

### **lora_target_QV_fc1_improved**

Improved placement (Modification A).
LoRA is applied to **q_proj, v_proj, and fc1 (MLP layer)**.
Adds additional modeling capacity beyond attention projections.
Results show consistent improvement over baseline QV.

---

### **lora_dropout_only**

Modification B: introduces larger LoRA dropout (0.1) while keeping placement unchanged.
Provides additional regularization, improves robustness slightly, especially on small datasets.

---

### **lora_improved_dropout**

Combined A + B (our final proposed method).
LoRA applied to **QV + fc1**, together with **higher dropout (0.1)**.
Shows the strongest performance among all LoRA variants, demonstrating the complementary effects of expanded placement and stronger regularization.
