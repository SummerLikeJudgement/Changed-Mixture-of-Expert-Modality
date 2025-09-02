import os

from torch import optim, nn
import torch
from utils.data_util import load_data, generate_kfolds_index
from sklearn.metrics import accuracy_score
import numpy as np
from GSR.model import CNN1D

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epoch, device, fold_idx):
    print(f"===开始训练第{fold_idx}折===")
    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []
    best_acc = 0

    for epoch in range(num_epoch):
        # 训练
        model.train()
        train_loss = 0.0
        train_pred, train_true = [], []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pred.append(torch.argmax(output, dim=1).cpu())
            train_true.append(target.cpu())

        train_loss /= len(train_loader)
        pred, true = torch.cat(train_pred), torch.cat(train_true)
        pred = pred.detach().cpu().numpy()
        true = true.detach().cpu().numpy()
        train_acc = accuracy_score(true, pred)
        # 验证
        model.eval()
        val_loss = 0.0
        valid_pred, valid_true = [], []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

                valid_pred.append(torch.argmax(output, dim=1).cpu())
                valid_true.append(target.cpu())
        val_loss /= len(val_loader)
        valid_pred, valid_true = torch.cat(valid_pred), torch.cat(valid_true)
        valid_pred = valid_pred.detach().cpu().numpy()
        valid_true = valid_true.detach().cpu().numpy()
        valid_acc = accuracy_score(valid_true, valid_pred)
        # 记录指标
        train_losses.append(train_loss)
        valid_losses.append(val_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        # 保存最佳模型
        if valid_acc > best_acc:
            best_acc = valid_acc
            model_save_dir = f"./pt/best_of_fold{fold_idx}.pth"
            os.makedirs(os.path.dirname(model_save_dir), exist_ok=True)
            torch.save(model.state_dict(), model_save_dir)

        if (epoch + 1)% 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epoch}"
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                  f"Valid Loss: {val_loss:.4f}, Valid Acc: {valid_acc:.4f}")
    return {
        'best_acc':best_acc,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'valid_losses': valid_losses,
        'valid_accs': valid_accs
    }

def main(data_dir = "", num_epoch = 50):
    k_folds = 5
    label_converter = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
    batch_size = 32
    folds_acc = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfolds_index = generate_kfolds_index(data_dir, k_folds)

    for fold_idx in range(k_folds):
        print(f"==={fold_idx+1}/{k_folds}折训练===")
        # 加载数据
        train_files, valid_files = kfolds_index[fold_idx]
        train_loader , valid_loader, dist = load_data(train_files, valid_files, label_converter, batch_size, num_workers=1)
        print(f"训练集{len(train_loader.dataset)}"
              f"验证集{len(valid_loader.dataset)}")
        # 初始化模型
        model = CNN1D(num_classes=len(label_converter)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        # 训练模型
        result = train_model(model, train_loader, valid_loader, criterion, optimizer, num_epoch=num_epoch, device=device, fold_idx=fold_idx)
        # 记录结果
        folds_acc.append(result['best_acc'])

    # 输出结果
    print("===交叉验证结果===")
    for i, acc in enumerate(folds_acc):
        print(f"第{i+1}折准确率{acc:.4f}")
    mean_acc = np.mean(folds_acc)
    std_acc = np.std(folds_acc)
    print(f"平均准确率{mean_acc:.4f}+-{std_acc:.4f}")
    print(f"最佳准确率{max(folds_acc):.4f}")


