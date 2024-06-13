### Soruce: https://www.kaggle.com/code/matthewjansen/transfer-learning-brain-tumor-classification
### DataSet: https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c

### Edit Version: https://www.kaggle.com/code/aminaslam/transfer-learning-brain-tumor-classificat-b4c34f
-----------------------------------
<a id=toc></a>
<h1 style="padding: 35px;color:white;margin:10;font-size:200%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://i.postimg.cc/T1D2yGny/167.jpg); background-size: 100% auto;background-position: 0px 0px; 
"><span style='color:white;'>Transfer Learning | Brain Tumor Classification</span></h1>

<center>
    <figure>
        <img src="https://cdn-images-1.medium.com/max/800/1*f1sodi17fNcObGBmIKgGGQ.gif" alt ="Brain Tumor MRI" style='width:55%;'>
        <figcaption>
            Source: <a href="https://www.ai-tech.systems/brain-tumor-detection/">AI Technology & Systems | Brain Tumor Detection</a>
        </figcaption>
    </figure>
</center>

## 🎯 Objective
The objective for this notebook is to explore the usage of transfer learning models, **namely EfficientNet V2 B0 and ViT-B16**, along with **ensembling methods** for solving the task of **classifying multiple types of brain tumors in patients**. The inspection of model performances post-training and selection process is also explored within this notebook. To you, the notebook visitor: I hope you find the contents insightful/useful. 


## 📁 Dataset
This dataset consists of a private collection of T1, contrast-enhanced T1, and T2 magnetic resonance images separated by brain tumor type. The images were collected without any type of marking or patient identification, interpreted by radiologists and provided for study purposes. The images are separated by astrocytoma, carcinoma, ependymoma, ganglioglioma, germinoma, glioblastoma, granuloma, medulloblastoma, meningioma, neurocytoma, oligodendroglioma, papilloma, schwannoma and tuberculoma.

**For more information check the following:**
> - [Kaggle | Brain Tumor MRI Images 44 Classes](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c?sort=published)
> - [Dataset Author | Fernando Feltrin](https://www.kaggle.com/fernando2rad)

## 🧠 What Are Brain Tumors?

<center>
    <figure>
        <img src="https://my.clevelandclinic.org/-/scassets/images/org/health/articles/6149-brain-tumor" alt ="Brain Tumor" style='width:40%;'>
        <figcaption>
            Source: <a href="https://my.clevelandclinic.org/health/diseases/6149-brain-cancer-brain-tumor">Cleveland Clinic | Brain Tumor Illustration</a>
        </figcaption>
    </figure>
</center>

A brain tumor is a cancerous or non-cancerous mass or growth of abnormal cells in the brain. Nearby locations include nerves, the pituitary gland, the pineal gland, and the membranes that cover the surface of the brain. Brain tumors that begin in the brain are called primary brain tumors. Sometimes, cancer spreads to the brain from other parts of the body. These tumors are known as secondary brain tumors, also called metastatic brain tumors.

### <u>Symptoms</u>
General signs and symptoms caused by brain tumors may include:
> - Headache or pressure in the head that is worse in the morning
> - Headaches that happen more often and seem more severe
> - Headaches that are sometimes described as tension headaches or migraines
> - Nausea or vomiting
> - Eye problems, such as blurry vision, seeing double or losing sight on the sides of your vision
> - Losing feeling or movement in an arm or a leg
> - Trouble with balance
> - Speech problems
> - Feeling very tired
> - Confusion in everyday matters
> - Memory issues
> - Having trouble following simple commands
> - Personality or behavior changes
> - Seizures, especially if there is no history of seizures
> - Hearing problems
> - Dizziness or a sense that the world is spinning, also called vertigo
> - Feeling very hungry and gaining weight

**For more information see the following:**
> - [Mayo Clinic | Brain tumor](https://www.mayoclinic.org/diseases-conditions/brain-tumor/symptoms-causes/syc-20350084)
> - [Cleveland Clinic | Brain Cancer (Brain Tumor)](https://my.clevelandclinic.org/health/diseases/6149-brain-cancer-brain-tumor)

<hr>

## Table of contents
- [1 | Dataset Exploration](#1)
   > - [Get image paths with glob](#1.1)
   > - [View the number of images present in the dataset](#1.2)
   > - [Create Pandas DataFrames for paths and labels](#1.3)
   > - [Load & View Random Sample Image](#1.4)
   > - [View Multiple Randomly Selected Samples](#1.5)
   > - [View Train Labels Distribution](#1.6)
   > - [Discard Insufficient Sample Classes](#1.7)
  
- [2 | Data Preprocessing: Building An Input Data Pipeline](#2)
   > - [Create Train & Validation Splits](#2.1)
   > - [View New Train & Validation Labels Distribution](#2.2)
   > - [Create an Image Data Augmentation Layer](#2.3)
   > - [Create Input Data Pipeline w. tf.data API](#2.4)
   
- [3 | Transfer Learning Model: EfficientNet V2 B0](#3)
   > - [TensorFlow Hub](#tfhub)
   > - [Get EfficientNet From TensorFlow Hub](#3.1)
   > - [Define EfficientNet Model](#3.2)
   > - [Train EfficientNet Model](#3.3)

- [4 | Transfer Learning Model: Vision Transformer (ViT)](#4)
   > - [Get Vision Transformer Model](#4.1)
   > - [Define Vision Transformer Model](#4.2)
   > - [Train Vision Transformer Model](#4.3)
   
- [5 | Ensembling via Averaging](#5)
   > - [What is Ensembling via Averaging?](#5.1)
   > - [Simple Average Ensembling](#5.2)
   > - [Weighted Average Ensembling](#5.3)
   > - [Geometric Mean Ensembling](#5.4)

- [6 | Performance Evaluation](#6)
   > - [View Model Histories](#6.1)
   > - [Plot Confusion Matrices](#6.2)
   > - [Inpsect Classification Reports](#6.3)
   > - [Record Classification Metrics](#6.4)
   > - [Trade-offs: Inference Time vs. Performance](#6.5)
   
- [Conclusion](#conclusion)

<hr>
