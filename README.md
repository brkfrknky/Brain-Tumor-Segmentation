# Brain-Tumor-Segmentation

This project is an open-source software endeavor encompassing a suite of tools and algorithms for brain tumor segmentation in the field of medical imaging. Brain tumors are a significant concern in the healthcare sector, and accurate segmentation is crucial for early diagnosis and effective treatment. This project has been developed with the aim of automatically identifying and segmenting brain tumors using image processing and deep learning techniques.

# Features

Image Processing Algorithms:

![image](https://github.com/brkfrknky/Brain-Tumor-Segmentation/assets/76915533/ba15dba1-3a1f-4066-ae1c-729e6603adc1)

Deep Learning Models: 

In the project, the specifically developed U-Net architecture for biomedical images has been utilized. 
While the standard U-Net model is designed for 512x512x3 images, the data used in this project has been set to 128x128x2. Accordingly, modifications have been made to the U-Net model based on these input values.

Dataset: 

The project includes the BraTS MRI brain tumor image dataset, meticulously prepared by Sicas for use in competitions during the learning and testing phases. This dataset comprises 369 MRI scans and is designed for professional utilization in competitions.

# Results 

![image](https://github.com/brkfrknky/Brain-Tumor-Segmentation/assets/76915533/da66599a-3417-4ab5-b9fa-418b7ea2a9af)

![image](https://github.com/brkfrknky/Brain-Tumor-Segmentation/assets/76915533/33b26b64-e7ae-4fd5-996c-a0a0aacd4937)

# Metric Results 
Accuracy: 0.9922 | Validation Accuracy: 0.9909
Jaccard: 0.9783 | Validation Jaccard: 0.9764
Loss: 0.0202 | Validation Loss: 0.0239
Precision: 0.9947 | Validation Precision: 0.9932
Recall: 0.9890 | Validation Recall: 0.9879
