# Artist identification from paintings

Can a model identify an artist from a painting alone? We fine-tuned MobileNetV3-Small on paintings by 379 artists and compared two ways to predict the artist ID. The first uses the network's own classification head. The second uses a linear support vector machine (SVM) trained on the same network's 256-dimensional image embeddings. Both models use only image pixels at prediction time.

[Training and evaluation notebook](Painter_CNN_Embeddings_SVM_PyTorch.ipynb) · [Project report](Team_32_Artist_Identification_Report.docx) · [Paper presentation](CNN_Embeddings_SVM_Paper_Explainer_10_Slides.pptx)

## Results

The saved experiment matched 5,787 painting images to artist IDs in `train_info.csv`. Keeping artists with at least five paintings left 4,089 images. A stratified split assigned 3,271 images to training and 818 to validation.

| Classifier | Correct predictions | Validation accuracy |
| --- | ---: | ---: |
| MobileNetV3-Small classification head | 226 / 818 | 27.63% |
| Standardized CNN embeddings + linear SVM | 223 / 818 | 27.26% |

The SVM scored 0.2044 macro-F1 when each artist received equal weight. For 281 of the 379 artists, validation contained only one or two paintings. The same validation set guided checkpoint and SVM selection. An independent test score is still needed.

## Run the notebook

1. Obtain the painting images separately. To reproduce the saved split, place the original **5,787-image subset** under `train_1_extracted/` beside the notebook. The notebook matches filenames recursively against the included CSV. It can also extract a separately supplied `train_1.zip`. The full archive contains 11,025 images and yields a different experiment.
2. Open the notebook in Jupyter with a Python kernel. The saved run used Python 3.12. Install the dependencies listed in section 2 by uncommenting and running its `%pip install` line, then restart the kernel. The first model build downloads ImageNet weights unless they are already cached.
3. Run the cells in order. Training writes the CNN checkpoint, SVM, scaler and artist label encoder to the project directory. Use a separate working copy to preserve the bundled models when retraining.
4. To predict the artist of another painting in the same session, set `SAMPLE_IMAGE` to its path and run section 23. The prediction is one of the dataset's artist IDs.

## Team 32

| Member | Student ID |
| --- | --- |
| Arham Mayur Burad | PES2UG24AM025 |
| K C Nandan | PES2UG24AM072 |
