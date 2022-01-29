# FasterRCNN

FasterRCNN is a [object-detection](https://en.wikipedia.org/wiki/Object_detection)
neural network (think bounding boxes). Training and testing scripts can be found in 
the directory:
`/data/aneurysm/tranch29/fasterrcnn` (to be uploaded to Github at a later date).

For now, feel free to copy the directory recursively to your personal directory with `cp -r [path] [path destination]`


## Dataset

Before we dive into FasterRCNN, the dataset needs to be addressed.

The current dataset for training and testing with FasterRCNN consists of 61 CT scans in
.png format. They can be considered a single "slice" of a patient's full torso, for which
the ascending and descending aorta can be most clearly seen.

These images were collected and annotated by Ronald Yang from the much bigger [TCGA-LUAD](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga) CT scan dataset. 

In this section, we'll focus on the .png images and their respective bounding box annotations
for FasterRCNN training and testing.

## Dataset Source 

The dataset can be found on Lambda here: `/data/aneurysm/tranch29/datasets/base`.

Notice that there are a few subdirectories in this path. Please ignore them for now.

The files/directories of interest are the following:

- the `_annotations.coco.json` file, which contains data on where the bounding boxes are in
each image. This is in [COCO](https://cocodataset.org/#home) format.

- the `images` directory, which contains all the images.

## Dataset Script 

Back in the FasterRCNN directory, the best place to start would be in the dataloader.
The dataloader class can be found in the `dataset.py` file. We'll attempt to walk
through the code as to get a good feel for how PyTorch dataloaders work.

Datasets objects are important because they handle the process of loading and processing data for dataloaders (which will be explained later).
Consider them the "middle-man" between raw data and the formatted data that machine learning models require as input.

Typically, dataloaders return the original image in as a [PyTorch tensor](https://pytorch.org/docs/stable/tensors.html) (think 2D array in the case of a grayscale image) and
a target object, which contains the ground truth/the "goal" of the machine learning model.

## Dataset Implementation Walkthrough

In this section, I will highlight important sections of the `dataset.py` file and try to explain them as best as I can.

### Seeds and Reproducibility

-- Line 23, dataset.py --

```Python
random.seed(0)
np.random.seed(0)
imgaug.random.seed(0)
torch.manual_seed(0)
```

After all the imports, you'll notice a section of code setting the random seeds of the different libraries to the value `0`.
This is because some pre-processing steps apply a random effect to images. Random seeds are set to some constant (in this case, 0) so that these random effects are reproducible.

### Augmentations

-- Line 27, dataset.py --

```Python
# For training
aug_transform = A.Compose([
	A.RandomRotate90(p=0.5),
	A.Blur(blur_limit=3, p=0.2),
	A.RandomBrightnessContrast(brightness_limit=0.8, contrast_limit=0.8, p=0.5),
	ToTensorV2()],
	bbox_params=A.BboxParams(format='pascal_voc', label_fields=['label']))

default_transform = A.Compose([
	ToTensorV2()],
	bbox_params=A.BboxParams(format='pascal_voc', label_fields=['label']))
```

The above code details two different augmentations a dataloader can be set to have. One is the `aug_transform` augmentations, which randomly
rotates an image by 90 degrees, blurs it by a random intensity, and adjusts its brightness and contrast by some amount. Note that, if applicable,
these transformations are also applied to the bounding boxes (such as the rotating transformation).

The other is the `default_transform`, effectively does nothing and leaves the original image and bounding boxes as is. 

Note that both augmentations share a `ToTensorV2` transformation. This converts the read image input (which would be a numpy array) into a PyTorch tensor. 
This is important because most PyTorch machine learning models need their images in tensor format rather than numpy format.

Feel free to make your own augmentations. Explore what options you have in the
[Albumentations library documentation](https://albumentations.ai/docs/api_reference/full_reference/).

### Main Dataset Class

-- Line 120, dataset.py --- 

```Python
class AneurysmDataset(dset.CocoDetection):
	def __init__(self, root, annFile, transform=None):
		super().__init__(root, annFile, transform)

	def __getitem__(self, index):
		coco = self.coco
		img_id = self.ids[index]

		# Get annotations
		ann_ids = coco.getAnnIds(imgIds=img_id)
		annotations = coco.loadAnns(ann_ids)

		# Filter out degenerate bounding boxes with 0 width or height
		annotations = [a for a in annotations
					   if a['bbox'][2] != 0 or a['bbox'][3] != 0]

		boxes = [a['bbox'] for a in annotations]
		area = []

		if len(boxes) > 0:
			for i in range(len(boxes)):
				area.append(boxes[i][2] * boxes[i][3])

				boxes[i] = [boxes[i][0], boxes[i][1],
							boxes[i][0] + boxes[i][2],
							boxes[i][1] + boxes[i][3]]
				boxes[i] = torch.LongTensor(boxes[i])

		# Labels
		labels = [a['category_id'] for a in annotations]
		labels = torch.LongTensor(labels)

		iscrowd = torch.zeros(len(labels), dtype=torch.int64)

		# Get image
		img_name = coco.loadImgs(img_id)[0]['file_name']
		img_path = os.path.join(self.root, 'images', img_name)
		img = read_img(img_path, mask=False)

		# Get mask images (for U-Net)
		mask_path = os.path.join(self.root, 'masks', img_name)
		mask = read_img(mask_path, mask=True)

		# Transforms
		transformed = self.transform(image=img, mask=mask, bboxes=boxes, label=labels)
		img = transformed['image']
		mask = transformed['mask']
		boxes = transformed['bboxes']

		# Convert box to tensors
		for i in range(len(boxes)):
			if type(boxes[i]) == torch.FloatTensor:
				boxes[i] = torch.FloatTensor(boxes[i])
		boxes = torch.stack(boxes)

		# Build target
		target = {}
		target['mask'] = mask
		target['boxes'] = boxes
		target['labels'] = labels
		target['image_id'] = torch.tensor([img_id])
		target['area'] = torch.LongTensor(area)
		target['iscrowd'] = iscrowd

		return img, target, img_name
```

This is the Dataset class. To instantiate it, three things are needed:
1. The root directory of the dataset is passed in (ex: `/data/aneurysm/tranch29/datasets/base`).
2. The path of the annotation file (ex: `/data/aneurysm/tranch29/datasets/base/_annotations.coco.json`).
3. And the augmentation (ex: `default_aug`, as previously mentioned). 

You can treat it as an array of data entries. For example, if you wanted to access the first data entry, which would include the image tensor, the target object, and the image name,
just use Python's bracket notation.

Ex:
```Python
myDataset = AneurysmDataset('/data/aneurysm/tranch29/datasets/base', '/data/aneurysm/tranch29/datasets/base/_annotation.coco.json', default_aug)
entry = myDataset[0]
print(entry)
# Will print tuple containing image, target object, and image name
```

### Dataloader

Dataloaders are essential for training machine learning models. They are wrappers for the Dataset class and allow
for easier access to the data by allowing to put them in batches for model training.

--- Line 187, dataset.py --- 

```Python
def get_loaders(data_path, anno_path, isKFold=False, n_splits=5, shuffle=True, random_state=0, debug=False):
    master_set = AneurysmDataset(data_path, anno_path, transform=default_transform)
    # master_set = _coco_remove_images_without_annotations(master_set)

    if isKFold:
        # Create KFolds from master set
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        fold_indices = kf.split(master_set)

        train_test_folds = []

        for fold, (train_indices, test_indices) in enumerate(fold_indices):
            # Debugging
            if debug:
                print('FOLD:', fold)
                print('TRAINING: ', end='')
                print(train_indices)
                print('TESTING: ', end='')
                print(test_indices)
                print()

            train_subset = Subset(master_set, train_indices)
            test_subset = Subset(master_set, test_indices)
            train_loader = DataLoader(train_subset, batch_size=1, num_workers=1, collate_fn=collate_fn)
            test_loader = DataLoader(test_subset, batch_size=1, num_workers=1, collate_fn=collate_fn)

            train_test_folds.append([train_loader, test_loader])
        return train_test_folds

    else:
        # Make dataloader from one single dataset
        dataloader = DataLoader(master_set, batch_size=1, num_workers=1, shuffle=True,
                                  collate_fn=collate_fn)
        return [dataloader]
```

Dataloaders are created in this function. However, it does a little bit more than make loaders - it also
creates loaders for [k-fold cross-validation](https://machinelearningmastery.com/k-fold-cross-validation/).

Further details on k-fold cross-validation can be read in the link above. It is important in small
datasets such as the one we have in this project - for both training and testing.

## Training

The current training script can be found in `train.py` in the FasterRCNN directory. 

At the moment, many of the parameters are hard-coded in the script, so it may be easier to implement an
argument parsing function in the future.

For now, however, I will go over some of the change-able hard-coded parameters in the training script.

--- Line 5, train.py --- 

```Python
...
train_path = '../datasets/base'
train_anno = '../datasets/base/_annotations.coco.json'
...
```

These variables define where the dataset and annotation file is located. The k-fold dataloaders will be built
from it. If you have another dataset with COCO annotations that you would like
to train on, feel free to change these variables as needed.

--- Line 16, train.py ---

```Python
...
lr = 0.001
lr_func = lambda epoch: 0.97 ** epoch
momentum = 0.9
weight_decay=0.0005
epochs = 100
...
```

- `lr` defines the [learning rate](https://en.wikipedia.org/wiki/Learning_rate) of the training loop. Typically,
the learning rate is kept at a pretty low range, but definitely try to play around with it if you are
training a new model.

- `lr_func` defines the learning rate function, which is a function of the current training [epoch](https://www.jigsawacademy.com/blogs/ai-ml/epoch-in-machine-learning).
Currently, this function is $learning\_rate = 0.97^{current\_epoch}$

```{note}
You can opt out of using this custom learning rate function altogether and use
PyTorch's ReduceLROnPlateau function, which may be a better choice in some cases.
Simply replace the scheduler variable on line 29 with [torch.optim.lr_scheduler.ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) and whatever the appropriate parameters are.
```

- `momentum` defines the [momentum](https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d) value, which would later be passed into PyTorch's [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) implementation for backpropagation.

- `weight_decay` defines a value than reduces the values of the model weights. In simple terms, it
prevents one weight from having too large of a value, which usually leads to overfitting.

- `epochs` defines how many epochs the model will train on in each fold. More can be read about it [here](https://www.jigsawacademy.com/blogs/ai-ml/epoch-in-machine-learning).

--- Line 24, train.py ---

```Python
...
    # Reset model parameters
	model.load_state_dict(torch.load('./initial_weights.pth'))
...
```

Here defines the initial weights from which each fold will start at.
The `initial_weights.pth` weight file contains weights with random values.
However, if you would like to start from pre-trained weights, swap out the
`torch.load` function's parameter with another valid weight file that matches
the FasterRCNN architecture.

--- Line 75, train.py ---

```Python
...
            if train_iters_total % 50 == 0:
                print(f'Iteration {train_iters_total} loss: {train_loss}')
            train_loss_total += train_loss
...
```

Here defines how often the training loss will be printed. The frequency of the print statement
can be adjusted in the conditional. This was used to record training loss for each
fold, piping the print output into a log file.

--- Line 79 ---

```Python
...
        # if epoch % 20 == 0:
        #     torch.save(model.state_dict(), f'./fold{fold+1}_epoch{epoch}_frcnn_trained.pth')

...
```

Here defines how often the model weights are saved. Uncomment these lines and change
the conditional to reflect how often you would want model weights to be saved. Note
that the model weights are saved at the end of the training process anyways and saving
more weights in between can slow down training time, so uncomment if you are, for example, investigating overfitting.

--- Line 92 ---
```Python
...
    torch.save(model.state_dict(), f'./fold{fold+1}_epoch{epoch}_frcnn_trained.pth')
...
```

Lastly, here is the path of where model weights will be saved, according to fold and
epoch. Adjust accordingly.

To start training, just run the `train.py` file. You might want to pipe the output to a text file, which contains the model's loss for each epoch, to process later:

```md
[path-to-python-env-binary] train.py > training_log.txt
```

If this is unclear, please reference the [environment setup page](environment-setup.md).

## Testing

Testing is done in a separate script in the same directory named `test.py`.

In this section, I will walk through some of the parameters you can change and what the output means. I suspect the first script to become obsolete (as the team gets more
data and change/add post-processing steps), but it is good to see how to load testing images into a FasterRCNN model.

### Changeable Parameters

--- Line 103 ---
```Python
...
	train_path = '../datasets/base'
	train_anno = '../datasets/base/_annotations.coco.json'
...
```

Here is the location of the COCO dataset being loaded in. Folds will be made
automatically in the testing function - feel free to trace back the code if you
want, but it calls the `get_loader` function [referenced before](###Dataloaders).

--- Line 118 ---
```Python
...
	iou_threshold = 0.5 # IOU threshold needed for a prediction to be correct
	confidence_threshold = 0.4 # Will filter out all boxes below this threshold
...
```

Here defines the IOU threshold and the confidence threshold:
- The `iou_threshold` defines the [intersection-over-union/Jaccard index](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/) a bounding box
needs to achieve with respect to the ground truth to be considered a true positive.
- The `confidence_threshold` defines the confidence a model's output needs to have to even be considered a true positive or false positive. I.E. all bounding boxes with a confidence
value below this threshold will be filtered out before any analysis.

--- Line 118 ---
```Python
...
	         weights_file = f'paths/crossfold/cv60/fold{fold+1}_epoch60_frcnn_trained.pth'
...
```

Here defines the weight files. Change it to fit whatever format you have saved the weights for each fold.



### Output

The testing script outputs a series of statistics for each fold, then the same series of statistics averages across all the folds.

Here are some of what those terms mean:
* TP - stands for true positive, the amount of times a bounding box correctly labeled an aorta.
* FP - stands for false positive, the amount of times a bounding box was wrong, either with respect to its IOU or its label.
* FN - stands for false negative, the amount of times an aorta was NOT labeled. I.E. times the model missed an aorta.
```{note}
The terms below all reference the [mean average precision score](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173). It is a common metric used
when evaluating object detection models.
```
* AAP - stands for the mean average precision of ascending aortas only.
* DAP - stands for the mean average precision of descending aortas only.
* mAP - stands for the mean average precision of all aortas regardless of label.
