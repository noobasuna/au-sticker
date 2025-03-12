# Au-Sticker

**Causal Placement of Adversarial Stickers Against Emotion Leakage**

![ARCH2](https://github.com/user-attachments/assets/4ff34eb8-89b8-433e-acd1-60fc3bd308e1)

## Overview

Securing micro-expressions against leakage is crucial for privacy, as these subtle facial movements convey genuine emotions and are inherently personal. This study aims to protect micro-expression data from potential adversarial attacks, ensuring the preservation of individuals' privacy and preventing unauthorized access or misuse of sensitive emotional information. Unlike traditional methods, which often require training and extensive access to models, this research introduces a novel post-hoc method that does not require additional training. We focus on physical adversarial attacks in micro-expression recognition, involving intentional manipulation of visual cues to deceive recognition systems and protect individual emotional privacy. Our approach leverages a causal discovery algorithm to identify causal relationships between facial parts, enabling rapid identification of the optimal locations for adversarial patches in frames with triggered micro-expressions. This method exhibits a more consistent attack success rate than randomly placed adversarial stickers, demonstrating effective generalization across different emotions, stickers, and models. Particularly relevant in scenarios with restricted access to the model, our technique requires only a single interaction during the attack process, highlighting its efficiency and minimal need for querying the target model. The proposed method effectively balances privacy protection with high generalization capability, setting a new standard for defending against adversarial threats in micro-expression recognition.


## Steps to Generate Adversarial Images

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/au-sticker.git
   cd au-sticker
   ```
   
2. Install the required dependencies:

  ```bash
  pip install -r requirements.txt
  ```

3. Run the main script to generate adversarial images:

  ```bash
  python main.py
```

### Configuration

You can customize the following parameters:
- Image size
- Sticker types and sizes
- Model architecture
- Training parameters
- Output directories

## Citation
If you find this work useful, please cite our paper:

```bibtex
@INPROCEEDINGS{10889387,
  author={Tan, Pei-Sze and Rajanala, Sailaja and Tan, Yee-Fan and Pal, Arghya and Tan, Chun-Ling and Phan, Raphaël C.-W. and Ong, Huey-Fang},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Post-Hoc Adversarial Stickers Against Micro-Expression Leakage}, 
  year={2025},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10889387}
}

## Contact
For any inquiries, please contact at tan.peisze@monash.edu.


