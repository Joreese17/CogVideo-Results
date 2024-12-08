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
   git clone https://github.com/yourusername/cogvideo-disney-finetuning.git
   cd cogvideo-disney-finetuning
   ```
   
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure multi-GPU settings**:
    - Update accelerate_config.yaml for your system.
  
4. **Start fine-tuning**:
   ```bash
   python finetune/train_cogvideox_lora.py
   ```

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
