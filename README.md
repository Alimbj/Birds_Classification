
# Bird Species Classification using Pretrained ResNet-101

This project implements a bird species classification model using a ResNet-101 network pretrained on ImageNet. The model is fine-tuned to recognize 525 different bird species using a dataset of bird images. The final fully connected layer is adjusted for the number of classes, and training is performed only on this last layer to leverage the pretrained feature extraction capabilities of ResNet-101.

## Project Structure
- **train_dir, val_dir, test_dir**: Directories containing the training, validation, and testing images.
- **model_weights.pth**: Saved weights of the trained model.


## Code Overview

1. **Data Preparation**:
   - Images are loaded using `torchvision.datasets.ImageFolder`.
   - A transformation pipeline is defined to resize images to 224x224, convert them to tensors, and normalize them to ResNet-101's required format.

2. **Model Architecture**:
   - A pretrained ResNet-101 model is loaded, and its final fully connected layer is modified to output 525 classes, matching the dataset.

3. **Training and Evaluation**:
   - The model's final fully connected layer is unfreezed and fine-tuned on the bird dataset.
   - The rest of the layers are frozen to retain pretrained weights, optimizing training efficiency.
   - A `train_and_evaluate_model` function manages the training and validation process, storing loss and accuracy metrics for analysis.

4. **Plotting**:
   - After training, the loss and accuracy over epochs are plotted using Matplotlib.

5. **Saving the Model**:
   - The trained model’s state dictionary is saved to `model_weights.pth`.

## Usage

1. **Training the Model**:
   The main training and evaluation script initializes data loaders, prepares the ResNet-101 model, and trains it for a specified number of epochs.
   ```
   resnet, resnet_losses_train, resnet_losses_test, resnet_accuracies_test = train_and_evaluate_model(
       model=resnet,
       train_dataloader=train_dataloader,
       test_dataloader=valid_dataloader,
       loss_fn=loss_fn,
       optimizer=optimizer,
       n_epochs=10
   )
   ```
2. **Plotting Results**:
   - The code generates plots for training and test losses, as well as test accuracy over each epoch.

3. **Saving Model Weights**:
   - The final model weights are saved to `model_weights.pth` using `torch.save`.

## Execution Time
The script outputs the total execution time, providing insights into model training efficiency.
