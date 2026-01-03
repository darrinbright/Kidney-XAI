import os, cv2, itertools, argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from glob import glob
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import timm
from torch.amp import autocast, GradScaler

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 4
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

def load_kidney_dataset(data_dir, class_names):
    all_images = []
    all_labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        if os.path.exists(class_path):
            image_files = glob(os.path.join(class_path, '*.jpg')) + glob(os.path.join(class_path, '*.png'))
            for img_path in image_files:
                all_images.append(img_path)
                all_labels.append(class_idx)
        else:
            print(f"Warning: Class directory '{class_path}' not found")
    
    if len(all_images) == 0:
        print(f"ERROR: No images found in {data_dir}")
        return pd.DataFrame()
    
    df = pd.DataFrame({
        'image_path': all_images,
        'label': all_labels,
        'class_name': [class_names[label] for label in all_labels]
    })
    
    return df

def clean_dataset(df, max_workers=4):
    def check_image(image_path):
        try:
            img = Image.open(image_path)
            img.load()
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return True
        except Exception:
            return False
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(check_image, df['image_path']),
            total=len(df),
            desc="Checking image integrity"
        ))
    
    df_clean = df[results].reset_index(drop=True)
    return df_clean

def balance_training_set(df_train, target_samples_per_class=None):
    df_train_balanced = pd.DataFrame()
    
    if target_samples_per_class is None:
        min_class_count = df_train['label'].value_counts().min()
        target_samples_per_class = min_class_count
    
    for i in range(num_classes):
        class_samples = df_train[df_train['label'] == i]
        current_count = len(class_samples)
        target = target_samples_per_class
        
        if current_count == 0:
            continue
        elif current_count < target:
            oversampled = class_samples.sample(n=target, replace=True, random_state=42)
        else:
            oversampled = class_samples.sample(n=target, random_state=42)
        
        df_train_balanced = pd.concat([df_train_balanced, oversampled], ignore_index=True)
    
    return df_train_balanced

def initialize_model(model_name, num_classes, use_pretrained=True):
    if model_name == "mobilenet":
        model = models.mobilenet_v2(pretrained=use_pretrained)
        num_ftrs = 1280
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        return model
    
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        return model
    
    elif model_name == "efficientnet":
        model = timm.create_model('efficientnet_b1', pretrained=use_pretrained, num_classes=0)
        num_ftrs = model.num_features
        
        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        class EfficientNetWrapper(nn.Module):
            def __init__(self, backbone, classifier):
                super().__init__()
                self.backbone = backbone
                self.classifier = classifier
            
            def forward(self, x):
                try:
                    features = self.backbone.forward_features(x)
                except:
                    features = self.backbone.global_pool(self.backbone.forward_features(x))
                    features = self.backbone.flatten(features)
                    return self.classifier[2:](features)
                
                return self.classifier(features)
        
        return EfficientNetWrapper(model, classifier)
    
    else:
        raise ValueError(f"Invalid model name: {model_name}")

class MedicalImageAugmentation:
    def __init__(self, p=0.1):
        self.p = p
    
    def __call__(self, img):
        if torch.rand(1) < self.p:
            if torch.rand(1) < 0.5:
                img = self.add_brightness_variation(img)
        return img
    
    def add_brightness_variation(self, img):
        factor = 0.9 + torch.rand(1) * 0.2
        return torch.clamp(img * factor, 0, 1)

class KidneyDataset(Dataset):
    def __init__(self, df, transform=None, cache_images=True, max_corrupted_ratio=0.1):
        self.df = df
        self.transform = transform
        self.paths = self.df['image_path'].values
        self.labels = self.df['label'].values
        self.cache_images = cache_images
        self.image_cache = {}
        self.corrupted_indices = set()
        
        if cache_images:
            corrupted_count = 0
            for i in tqdm(range(len(self.paths)), desc="Caching images"):
                try:
                    img = self._load_image_safely(self.paths[i])
                    if img is not None:
                        self.image_cache[i] = img
                    else:
                        self.corrupted_indices.add(i)
                        self.image_cache[i] = None
                        corrupted_count += 1
                except Exception as e:
                    self.corrupted_indices.add(i)
                    self.image_cache[i] = None
                    corrupted_count += 1
            
            corrupted_ratio = corrupted_count / len(self.paths)
            if corrupted_ratio > max_corrupted_ratio:
                print(f"Warning: {corrupted_ratio:.2%} of images are corrupted. Disabling caching.")
                self.cache_images = False
                self.image_cache = {}
            else:
                print(f"Successfully cached {len(self.paths) - corrupted_count} images. {corrupted_count} corrupted images found.")
            
            if torch.cuda.is_available() and torch.cuda.memory_allocated() > torch.cuda.max_memory_allocated() * 0.8:
                print("High memory usage detected. Clearing cache.")
                self.image_cache.clear()
                torch.cuda.empty_cache()
    
    def _load_image_safely(self, image_path):
        try:
            img = Image.open(image_path)
            img.load()
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except (OSError, IOError) as e:
            return None
        except Exception as e:
            return None
        finally:
            if 'img' in locals():
                del img
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        if index in self.corrupted_indices:
            raise ValueError(f"Image at index {index} is corrupted")
        
        if self.cache_images and index in self.image_cache:
            X = self.image_cache[index]
            if X is None:
                raise ValueError(f"Image at index {index} is None in cache")
        else:
            X = self._load_image_safely(self.paths[index])
            if X is None:
                raise ValueError(f"Failed to load image at index {index}")
        
        y = torch.tensor(int(self.labels[index]))
        
        if self.transform:
            X = self.transform(X)
        
        return X, y

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, num_classes=4, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.class_weights = torch.tensor([1.2, 1.0, 1.5, 1.3])
    
    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            targets_one_hot = torch.zeros_like(inputs)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
            smooth_targets = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / inputs.size(1)
            ce_loss = -torch.sum(smooth_targets * torch.log_softmax(inputs, dim=1), dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        class_weights = self.class_weights.to(inputs.device)[targets]
        return (focal_loss * class_weights).mean()

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(train_loader, models, criterion, optimizers, epoch):
    for model in models.values():
        model.train()
    
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    
    individual_losses = {name: AverageMeter() for name in models.keys()}
    individual_accs = {name: AverageMeter() for name in models.keys()}
    
    accumulation_steps = 2
    
    for i, data in enumerate(train_loader):
        images, labels = data
        N = images.size(0)
        images = images.to(device)
        labels = labels.to(device)
        
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            outputs_ensemble = {}
            for name, model in models.items():
                outputs_ensemble[name] = model(images)
            
            total_loss = 0
            for outputs in outputs_ensemble.values():
                total_loss += criterion(outputs, labels)
        
        scaled_loss = total_loss / accumulation_steps
        scaler.scale(scaled_loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            for name in models.keys():
                scaler.unscale_(optimizers[name])
                torch.nn.utils.clip_grad_norm_(models[name].parameters(), max_norm=0.3)
                scaler.step(optimizers[name])
                scaler.update()
                optimizers[name].zero_grad(set_to_none=True)
        
        avg_loss = total_loss / len(models)
        
        for name, output in outputs_ensemble.items():
            individual_loss = criterion(output, labels)
            _, predicted = torch.max(output, 1)
            correct = predicted.eq(labels).sum().item()
            accuracy = correct / N
            
            individual_losses[name].update(individual_loss.item(), N)
            individual_accs[name].update(accuracy, N)
        
        weights = torch.tensor([0.25, 0.35, 0.40]).to(device)
        weights = weights / weights.sum()
        
        ensemble_output = torch.zeros_like(list(outputs_ensemble.values())[0])
        for j, (name, output) in enumerate(outputs_ensemble.items()):
            ensemble_output += weights[j] * output
        
        prediction = ensemble_output.max(1, keepdim=True)[1]
        correct = prediction.eq(labels.view_as(prediction)).sum().item()
        accuracy = correct / N
        
        train_acc.update(accuracy, N)
        train_loss.update(avg_loss.item(), N)
        
        del outputs_ensemble, total_loss, ensemble_output, prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if (i + 1) % 25 == 0 or (i + 1) == len(train_loader):
            print(f'[epoch {epoch}], [iter {i + 1} / {len(train_loader)}], [train loss {train_loss.avg:.5f}], [train acc {train_acc.avg:.5f}]')
    
    for name in models.keys():
        individual_train_losses[name].append(individual_losses[name].avg)
        individual_train_accs[name].append(individual_accs[name].avg)
    
    return train_loss.avg, train_acc.avg

def validate_epoch(val_loader, models, criterion, epoch):
    for model in models.values():
        model.eval()
    
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    
    individual_val_losses_epoch = {name: AverageMeter() for name in models.keys()}
    individual_val_accs_epoch = {name: AverageMeter() for name in models.keys()}
    
    all_predictions = []
    all_labels = []
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            N = images.size(0)
            images = images.to(device)
            labels = labels.to(device)
            
            outputs_ensemble = {}
            for name, model in models.items():
                outputs_ensemble[name] = model(images)
            
            total_loss = 0
            for outputs in outputs_ensemble.values():
                total_loss += criterion(outputs, labels)
            
            avg_loss = total_loss / len(models)
            
            for name, output in outputs_ensemble.items():
                individual_loss = criterion(output, labels)
                _, predicted = torch.max(output, 1)
                correct = predicted.eq(labels).sum().item()
                accuracy = correct / N
                
                individual_val_losses_epoch[name].update(individual_loss.item(), N)
                individual_val_accs_epoch[name].update(accuracy, N)
            
            weights = torch.tensor([0.25, 0.35, 0.40]).to(device)
            weights = weights / weights.sum()
            
            ensemble_output = torch.zeros_like(list(outputs_ensemble.values())[0])
            for j, (name, output) in enumerate(outputs_ensemble.items()):
                ensemble_output += weights[j] * output
            
            probabilities = torch.softmax(ensemble_output, dim=1)
            _, predicted = torch.max(ensemble_output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            correct = predicted.eq(labels.view_as(predicted)).sum().item()
            correct_predictions += correct
            total_predictions += N
            
            accuracy = correct / N
            
            val_acc.update(accuracy, N)
            val_loss.update(avg_loss.item(), N)
    
    for name in models.keys():
        individual_val_losses[name].append(individual_val_losses_epoch[name].avg)
        individual_val_accs[name].append(individual_val_accs_epoch[name].avg)
    
    print(f'[epoch {epoch}], [val loss {val_loss.avg:.5f}], [val acc {val_acc.avg:.5f}]')
    
    return val_loss.avg, val_acc.avg

def evaluate_model(model, val_loader, model_name):
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
    y_bin = label_binarize(all_labels, classes=range(num_classes))
    roc_auc = roc_auc_score(y_bin, all_probabilities, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score'],
        'roc_auc': roc_auc,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }

def test_time_augmentation(model, image, num_augmentations=5):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        pred = model(image.unsqueeze(0))
        predictions.append(torch.softmax(pred, dim=1))
    
    tta_transforms = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
    ]
    
    for i, transform in enumerate(tta_transforms[:num_augmentations-1]):
        with torch.no_grad():
            try:
                augmented = transform(image)
                pred = model(augmented.unsqueeze(0))
                predictions.append(torch.softmax(pred, dim=1))
            except Exception as e:
                predictions.append(predictions[0])
    
    final_pred = torch.mean(torch.stack(predictions), dim=0)
    return final_pred

def evaluate_ensemble(val_loader, weights, use_tta=True):
    for model in models_ensemble.values():
        model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            if use_tta:
                ensemble_outputs = []
                for i in range(images.size(0)):
                    single_image = images[i]
                    image_predictions = []
                    
                    for name, model in models_ensemble.items():
                        tta_pred = test_time_augmentation(model, single_image)
                        image_predictions.append(tta_pred)
                    
                    weighted_pred = torch.zeros_like(image_predictions[0])
                    for j, pred in enumerate(image_predictions):
                        weighted_pred += weights[j] * pred
                    
                    ensemble_outputs.append(weighted_pred)
                
                ensemble_output = torch.cat(ensemble_outputs, dim=0)
            else:
                outputs_ensemble = {}
                for name, model in models_ensemble.items():
                    outputs_ensemble[name] = model(images)
                
                ensemble_output = torch.zeros_like(list(outputs_ensemble.values())[0])
                for j, (name, output) in enumerate(outputs_ensemble.items()):
                    ensemble_output += weights[j] * output
            
            probabilities = ensemble_output
            _, predicted = torch.max(probabilities, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
    y_bin = label_binarize(all_labels, classes=range(num_classes))
    roc_auc = roc_auc_score(y_bin, all_probabilities, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score'],
        'roc_auc': roc_auc,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'per_class': report
    }

def calculate_ensemble_weights(individual_results):
    accuracies = []
    for name in ['mobilenet', 'resnet50', 'efficientnet']:
        if name in individual_results:
            accuracies.append(individual_results[name]['accuracy'])
        else:
            accuracies.append(0.5)
    
    weights = torch.softmax(torch.tensor(accuracies) * 10, dim=0)
    
    performance_threshold = 0.60
    for i, acc in enumerate(accuracies):
        if acc < performance_threshold:
            weights[i] = 0.1
    
    weights = weights / weights.sum()
    min_weight = 0.1
    weights = torch.maximum(weights, torch.tensor(min_weight))
    weights = weights / weights.sum()
    
    return weights.to(device)

def visualize_dataset_samples():
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    fig.suptitle('CT KIDNEY DATASET: Normal-Cyst-Tumor-Stone', fontsize=16, fontweight='bold', y=0.98)
    
    all_images = []
    for class_idx, class_name in enumerate(class_names):
        class_df = df_clean[df_clean['label'] == class_idx]
        if len(class_df) > 0:
            n_samples = min(6, len(class_df))
            class_samples = class_df.sample(n=n_samples, random_state=42)
            all_images.extend(class_samples['image_path'].tolist())
    
    while len(all_images) < 25:
        additional_samples = df_clean.sample(n=min(25-len(all_images), len(df_clean)), random_state=43)
        all_images.extend(additional_samples['image_path'].tolist())
    
    selected_images = all_images[:25]
    
    sample_count = 0
    for row_idx in range(5):
        for col_idx in range(5):
            if sample_count < len(selected_images):
                try:
                    img = Image.open(selected_images[sample_count])
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize((224, 224))
                    img_array = np.array(img)
                    
                    axes[row_idx, col_idx].imshow(img_array)
                    axes[row_idx, col_idx].axis('off')
                    
                    sample_count += 1
                
                except Exception as e:
                    axes[row_idx, col_idx].imshow(np.zeros((224, 224, 3), dtype=np.uint8))
                    axes[row_idx, col_idx].axis('off')
                    sample_count += 1
            else:
                axes[row_idx, col_idx].axis('off')
    
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.92, top=0.92, wspace=0.02, hspace=0.02)
    plt.savefig('results/dataset_samples.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_training_curves():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(range(1, len(train_accs) + 1), train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(range(1, len(val_accs) + 1), val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Ensemble Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    ax2.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss', linewidth=2)
    ax2.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Ensemble Training and Validation Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/TrainingValidation/ensemble_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_confusion_matrix(results, name):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(results['labels'], results['predictions'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'{name.upper()} Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'results/ConfusionMatrix/{name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_roc_curves(results, name):
    plt.figure(figsize=(10, 8))
    y_bin = label_binarize(results['labels'], classes=range(num_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], np.array(results['probabilities'])[:, i])
        roc_auc[i] = roc_auc_score(y_bin[:, i], np.array(results['probabilities'])[:, i])
    
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{name.upper()} ROC Curves', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/ROC/{name}_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def main(args):
    global df_clean, train_loader, val_loader, test_loader, models_ensemble, optimizers, schedulers
    global scaler, criterion, individual_train_losses, individual_train_accs
    global individual_val_losses, individual_val_accs, train_losses, train_accs, val_losses, val_accs
    global df_train, df_val, df_test, best_val_acc
    
    os.makedirs('results/ConfusionMatrix', exist_ok=True)
    os.makedirs('results/ROC', exist_ok=True)
    os.makedirs('results/TrainingValidation', exist_ok=True)
    os.makedirs('results/GradCAM', exist_ok=True)
    os.makedirs('results/LIME', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    print("Loading dataset...")
    df = load_kidney_dataset(args.data_dir, class_names)
    df_clean = clean_dataset(df, max_workers=4)
    
    y = df_clean['label']
    df_train, df_temp = train_test_split(df_clean, test_size=0.3, random_state=123, stratify=y)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=456, stratify=df_temp['label'])
    
    min_class_count = df_train['label'].value_counts().min()
    target_samples = min(500, min_class_count)
    
    if min_class_count == 0:
        df_train_balanced = df_train
    else:
        df_train_balanced = balance_training_set(df_train, target_samples_per_class=target_samples)
    
    df_train = df_train_balanced
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    training_set = KidneyDataset(df_train, transform=train_transform, cache_images=True)
    validation_set = KidneyDataset(df_val, transform=val_transform, cache_images=True)
    test_set = KidneyDataset(df_test, transform=test_transform, cache_images=True)
    
    print(f"Training dataset size: {len(training_set)}")
    print(f"Validation dataset size: {len(validation_set)}")
    print(f"Test dataset size: {len(test_set)}")
    
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    visualize_dataset_samples()
    
    models_ensemble = {}
    model_names = ['mobilenet', 'resnet50', 'efficientnet']
    
    for name in model_names:
        model = initialize_model(name, num_classes, True)
        model = model.to(device)
        models_ensemble[name] = model
    
    optimizers = {}
    for name, model in models_ensemble.items():
        if name == 'mobilenet':
            optimizers[name] = optim.AdamW(model.parameters(), lr=args.lr_mobilenet, weight_decay=args.weight_decay)
        elif name == 'resnet50':
            optimizers[name] = optim.AdamW(model.parameters(), lr=args.lr_resnet, weight_decay=args.weight_decay)
        else:
            optimizers[name] = optim.AdamW(model.parameters(), lr=args.lr_efficientnet, weight_decay=args.weight_decay)
    
    criterion = FocalLoss(alpha=1, gamma=args.focal_gamma, num_classes=4, label_smoothing=args.label_smoothing).to(device)
    scaler = GradScaler()
    
    schedulers = {}
    for name, optimizer in optimizers.items():
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=5
        )
        
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.6, patience=4, min_lr=1e-7, verbose=False
        )
        
        schedulers[name] = {'warmup': warmup_scheduler, 'main': main_scheduler}
    
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    individual_train_losses = {'mobilenet': [], 'resnet50': [], 'efficientnet': []}
    individual_train_accs = {'mobilenet': [], 'resnet50': [], 'efficientnet': []}
    individual_val_losses = {'mobilenet': [], 'resnet50': [], 'efficientnet': []}
    individual_val_accs = {'mobilenet': [], 'resnet50': [], 'efficientnet': []}
    
    best_val_acc = 0
    no_improve_count = 0
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(train_loader, models_ensemble, criterion, optimizers, epoch)
        val_loss, val_acc = validate_epoch(val_loader, models_ensemble, criterion, epoch)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        for name, scheduler_dict in schedulers.items():
            if epoch <= 5:
                scheduler_dict['warmup'].step()
            else:
                scheduler_dict['main'].step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
            torch.save({
                'mobilenet': models_ensemble['mobilenet'].state_dict(),
                'resnet50': models_ensemble['resnet50'].state_dict(),
                'efficientnet': models_ensemble['efficientnet'].state_dict(),
                'best_val_acc': best_val_acc,
                'epoch': epoch
            }, 'checkpoints/best_model.pth')
            print(f"New best model saved with validation accuracy: {best_val_acc:.5f}")
        else:
            no_improve_count += 1
        
        if no_improve_count >= args.patience:
            print(f'Early stopping at epoch {epoch} - no improvement for {args.patience} epochs')
            break
    
    print(f'\nTraining completed. Best validation accuracy: {best_val_acc:.5f}')
    
    save_training_curves()
    
    print("\nLoading best model for evaluation...")
    try:
        checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
        for name, model in models_ensemble.items():
            if name in checkpoint:
                model.load_state_dict(checkpoint[name])
        print("Best model loaded successfully")
    except FileNotFoundError:
        print("No saved model found. Using current model state.")
    
    print("\nEvaluating individual models...")
    individual_results = {}
    for name, model in models_ensemble.items():
        results = evaluate_model(model, test_loader, name)
        individual_results[name] = results
        
        print(f"\n{name.upper()}:")
        print(f"  Accuracy: {results['accuracy']:.5f}")
        print(f"  Precision: {results['precision']:.5f}")
        print(f"  Recall: {results['recall']:.5f}")
        print(f"  F1-Score: {results['f1_score']:.5f}")
        print(f"  ROC AUC: {results['roc_auc']:.5f}")
        
        save_confusion_matrix(results, name)
        save_roc_curves(results, name)
    
    print("\nEvaluating ensemble model...")
    dynamic_weights = calculate_ensemble_weights(individual_results)
    print(f"Dynamic Ensemble Weights: MobileNet={dynamic_weights[0]:.3f}, ResNet50={dynamic_weights[1]:.3f}, EfficientNet={dynamic_weights[2]:.3f}")
    
    ensemble_results = evaluate_ensemble(test_loader, dynamic_weights, use_tta=True)
    
    print(f"\nENSEMBLE MODEL (Test Set):")
    print(f"  Accuracy: {ensemble_results['accuracy']:.5f}")
    print(f"  Precision: {ensemble_results['precision']:.5f}")
    print(f"  Recall: {ensemble_results['recall']:.5f}")
    print(f"  F1-Score: {ensemble_results['f1_score']:.5f}")
    print(f"  ROC AUC: {ensemble_results['roc_auc']:.5f}")
    
    save_confusion_matrix(ensemble_results, 'ensemble')
    save_roc_curves(ensemble_results, 'ensemble')
    
    print("\nTraining complete! Results saved to ./results/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kidney-XAI: Ensemble Deep Learning for Kidney Disease Classification')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr_mobilenet', type=float, default=8e-5, help='Learning rate for MobileNet')
    parser.add_argument('--lr_resnet', type=float, default=4e-5, help='Learning rate for ResNet50')
    parser.add_argument('--lr_efficientnet', type=float, default=1.5e-4, help='Learning rate for EfficientNet')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for focal loss')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    
    args = parser.parse_args()
    main(args)
