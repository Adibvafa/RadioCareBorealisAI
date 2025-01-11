# RadioCare
![image](https://github.com/Adibvafa/RadioCareBorealisAI/assets/90617686/ec861f94-db64-42c0-b692-d5f2ba5fe737)
<br><br>

## Code
```python
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

# Load model
processor = BlipProcessor.from_pretrained("adibvafa/BLIP-MIMIC-CXR")
model = BlipForConditionalGeneration.from_pretrained("adibvafa/BLIP-MIMIC-CXR")

# Load data
image = 'chest-x-ray.jpg'
prompt = 'final report\nexamination: chest (pa and lat)\nindication: ___f with chest pressure, uri sx, voice change.'

# Process inputs
inputs = processor(
    images=Image.open(image), 
    text=prompt,
    return_tensors="pt"
)

# Generate radiology report
output = model.generate(**inputs, max_length=512)
report = processor.decode(output[0], skip_special_tokens=True)
```

```
### report will be like follows:
final report
examination : chest ( pa and lat )
indication : ___f with chest pressure, uri sx, voice change.
comparison : none
findings : pa and lateral views of the chest provided. there is no focal consolidation, effusion, or pneumothorax. the cardiomediastinal silhouette is normal. imaged osseous structures are intact. no free air below the right hemidiaphragm is seen.
impression : no acute intrathoracic process.
```

## Demo
![](radiocare_loop.gif)
<br><br>

## Introduction
Radiocare aims to develop a cutting-edge image-to-text model that generates accurate radiology reports and diagnoses for chest X-ray images. By leveraging the BLIP and Vision Transformer architectures, Radiocare seeks to streamline the diagnostic process, enabling faster and more accurate identification of health issues. This project addresses the critical need for timely and precise radiological assessments, especially in rural areas with limited access to healthcare. Ultimately, Radiocare strives to improve patient outcomes and bridge the gap in healthcare accessibility across Canada.
<br><br>

## Methods

### Data
Radiocare utilizes data from the MIMIC-CXR database on PhysioNet, consisting of a large collection of chest X-ray images and associated radiology reports. This dataset provides a comprehensive source of medical images essential for training and evaluating the model.
<br>
### Model Architecture
Radiocare employs the BLIP (Bootstrapped Language-Image Pre-training) model, which integrates the Vision Transformer (ViT) architecture with a text decoder. ViT processes images by dividing them into fixed-size patches, transforming these patches into high-dimensional vectors, and then embedding them into tokens. The self-attention mechanism in ViT captures global dependencies across patches, enhancing the model's understanding of the entire image. The text decoder translates these visual features into coherent radiology reports, enabling detailed and accurate diagnostics.
<br><br>

## Results
Radiocare's model can assess a chest X-ray in approximately 3 seconds, providing doctors with a 99% faster diagnostic process. Key performance metrics on "Findings" include:

- **Bert Socre**: 47.2
- **RadGraph**: 26.1

<br><br>

## Discussion
Radiocare represents a significant advancement in the field of medical diagnostics by leveraging state-of-the-art AI techniques to generate accurate and timely radiology reports from chest X-ray images. The integration of the BLIP model and Vision Transformer architecture enhances the diagnostic process, ensuring faster and more reliable results. By addressing the critical healthcare needs, especially in underserved rural areas, Radiocare has the potential to improve patient outcomes and bridge the gap in healthcare accessibility across Canada.
<br><br>

## Team Information
Radiocare is part of the Spring 2024 cohort of Borealis AI's "Let's SOLVE It" program. The project team includes:
- [Adibvafa Fallahpour](https://adibvafa.github.io/Portfolio/)
- [Archita Srivastava](https://www.linkedin.com/in/archita7/)
- [Mantaj Dhillon](https://www.linkedin.com/in/mantaj-dhillon/)
- [Grace Liu](https://www.linkedin.com/in/gracelliu/)
<br><br>

## Repository Structure
The repository is organized as follows:
- **data_modules/**: Contains data loading and preprocessing scripts.
- **evals/**: Includes evaluation scripts and metrics calculation.
- **models/**: Contains the different model architectures.
  - **blip/**: Final model implementation using BLIP and ViT.
  - **cnn/**: Convolutional neural network models.
  - **vit/**: Vision Transformer models.
- **utils/**: Utility functions for the project.
- **slurm/**: SLURM batch scripts for running jobs on a computing cluster.
<br><br>

## Citation

If you use this work in your research, please cite:

```
@misc {adibvafa_fallahpour_2024,
	author       = { Fallahpour, Adibvafa and Srivastava, Archita and Dhillon, Mantaj and Liu, Grace },
	title        = { BLIP-MIMIC-CXR },
	year         = 2024,
	url          = { https://huggingface.co/adibvafa/BLIP-MIMIC-CXR },
	doi          = { 10.57967/hf/3207 },
	publisher    = { Hugging Face }
}
```
