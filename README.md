# CogVideo Fine-Tuning for Disney-Style Video Generation

Welcome to the repository for our fine-tuning project using **CogVideo**, a state-of-the-art video generation model. This project adapts CogVideo to produce **Disney-style videos** characterized by vivid colors, intricate animations, and narrative-driven visuals. By leveraging multi-GPU distributed training and advanced fine-tuning techniques, we achieved significant improvements in video quality, aesthetic fidelity, and narrative consistency.

---

## 🚀 **Project Overview**

This project demonstrates how **domain-specific fine-tuning** of generative AI models can enhance their performance in specialized tasks. Using the **Disney Video Generation Dataset**, we refined CogVideo to generate videos that are stylistically accurate and contextually aligned with Disney's iconic animation style.

### Key Features:
- **Image-to-Video Generation**: Generate coherent video sequences from static images using diffusion-based methods.
- **Domain-Specific Fine-Tuning**: Tailored adaptations to replicate Disney's unique visual and narrative styles.
- **Multi-GPU Distributed Training**: Optimized training times with frameworks like [DeepSpeed](https://www.deepspeed.ai/) and [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/).
- **Enhanced Video Quality**: Improvements in temporal consistency, aesthetic fidelity, and prompt responsiveness.

---

## 🛠 **How It Works**

### Workflow:
1. **Dataset Preparation**: Processed the Disney Video Generation Dataset to map textual prompts to corresponding videos.
2. **Model Fine-Tuning**:
   - Leveraged multi-GPU acceleration for efficient training.
   - Applied hyperparameter tuning to optimize model performance.
3. **Evaluation**: Tested the fine-tuned model for stylistic accuracy, visual coherence, and responsiveness to complex prompts.


### Results:
- **Visual Coherence**: Smoother transitions and consistent character outlines.
- **Stylistic Accuracy**: Close alignment with Disney’s iconic aesthetic.
- **Prompt Fidelity**: Enhanced ability to generate contextually relevant videos.

---
## Prompts
# Two Detailed
A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions <br>
A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance
# Two Basic
Mickey the mouse talking to a goat on a stage in a black and white scene. <br>
A panda playing a guitar sitting on a stool in a bamboo forest to baby pandas

## Mickey Mouse Results Detailed Prompt Before Fine-tuning
[Before Finetuning](https://github.com/user-attachments/assets/88f87bfb-8f0a-4f23-8edc-eeb312ff3497)
## Mickey Mouse Detailed Prompt Results After Fine-tuning
[After Finetuning (100 Iterations)](https://github.com/user-attachments/assets/25ba047b-1391-44f3-9c17-25de84f922e0)

[After Finetuning (50 Iterations)](https://github.com/user-attachments/assets/5a256f47-87e9-48f5-b6bb-ed1a1c8fcf71)
## Mickey Mouse Basic Prompt Results Before Fine-tuning
[Before Finetuning](https://github.com/user-attachments/assets/e2d0eaa4-eb22-458c-96cc-c13c46979d99)
## Mickey Mouse Basic Prompt Results After Fine-tuning
[After Finetuning (50 Iterations)](https://github.com/user-attachments/assets/814a5e4b-c700-4741-90f8-ccbabad257ea)

[After Finetuning (10 Iterations)](https://github.com/user-attachments/assets/1f3914e7-d392-48f8-9f00-87b65194d784)
## Panda Detailed Prompt Results Before Fine-tuning
[Before Finetuning](https://github.com/user-attachments/assets/81d74160-f92f-4ca5-861c-103f1e0c3ddb)

## Panda Detailed Prompt Results After Fine-tuning
[After Finetuning (100 Iterations)](https://github.com/user-attachments/assets/18499591-78a4-418f-84d1-f828b92830b)

[After Finetuning (50 Iterations)](https://github.com/user-attachments/assets/17ec9cc9-3276-4072-9a59-d12c41ca6722)
## Panda Basic Prompt Results Before Fine-tuning
[Before Finetuning](https://github.com/user-attachments/assets/d6ceea78-e5ee-49e2-9ded-7e24c816d0bb)
## Panda Basic Prompt Results After Fine-tuning
[After Finetuning (50 Iterations)](https://github.com/user-attachments/assets/6a4d6dd1-a8cb-4bae-9ed8-1456459c94d9)

[After Finetuning (10 Iterations)](https://github.com/user-attachments/assets/a0440fbe-b823-4b4a-8eee-18b0a46938d1)



 

## 📊 **Baseline vs. Fine-Tuned Results**

| Metric                 | Baseline CogVideo          | Fine-Tuned CogVideo      |
|------------------------|----------------------------|--------------------------|
| **Temporal Consistency** | Smooth transitions         | Highly improved          |
| **Aesthetic Fidelity**  | Partial alignment          | Close to Disney quality  |
| **Prompt Responsiveness** | Limited success           | Handles complex prompts  |

**Examples**:
- **Baseline**: A panda playing guitar lacked detailed backgrounds.
- **Fine-Tuned**: Panda animations featured detailed bamboo forests and smoother transitions.

---

## 🔧 **Modifications**

- **Dataset Configuration**: Processed prompts and videos into compatible formats.
- **Multi-GPU Training**: Updated configuration files for multi-node setups.
- **Script Adjustments**: Integrated essential libraries (`peft`, `wandb`, `decord`) and resolved dataset path issues.
- **Hyperparameter Tuning**:
  - Learning rate: 0.0001
  - Batch size: 4
  - Epochs: 30

---

## 🖥️ **Setup Instructions**

### Prerequisites:
- Python 3.8 or later
- GPUs with CUDA support
- Installed frameworks: [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/), [DeepSpeed](https://www.deepspeed.ai/)

### Installation:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/THUDM/CogVideo
   cd CogVideo
   ```
   
2. **Load Required Modules**:
   ```bash
   module load mamba
   module load cuda-11.8.0-gcc-12.1.0
   ```

3. **Set Up Cache Directories**:
   - Replace `{your profile}` and `{your_dir}` with your actual profile and directory.

   ```bash
   export HF_HOME=/scratch/{your profile}/{your_dir}/cache
   export TMPDIR=/scratch/{your profile}/{your_dir}/cache
   export TRITON_CACHE_DIR=/scratch/{your profile}/{your_dir}/cache
   ```
  
4. **Create and activate a Conda environment**:
   ```bash
   mamba create -n CogEnv -c conda-forge
   source activate CogEnv
   ```

5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

6. **Set up the fine-tuning environment**:
   ```bash
   cd finetune
   git clone https://github.com/huggingface/diffusers.git
   cd diffusers
   pip install -e .
   ```

7. **Login to Hugging Face**:
   - Login to Hugging Face & Download the Disney Video Generation Dataset: Replace video-dataset-disney with your preferred local directory name.
   ```bash
   huggingface-cli login
   huggingface-cli download --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset --local-dir video-dataset-disney
   ```

8. **Modify the `accelerate_config_machine_single.yaml` File**:
   - Edit the `num_processes` value to match the number of GPUs available on your system (start with 1 if unsure).

9. **Verify Dataset File Names**:
   - Ensure the dataset file names match those referenced in the arguments of the `finetune_single_rank.sh` script.

---

### 📈 **Future Work**
- Address overfitting to specific training data.
- Minimize occasional artifacts in generated videos.
- Explore real-time video generation capabilities.

---

### 🤝 **Contributions**
We welcome contributions to enhance this project! Please feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss your ideas.

---

### 📝 **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
