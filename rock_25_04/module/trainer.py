import os
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from torch import amp
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None,
                num_epochs=10, device='cuda', log_dir="./runs/rock_classify", class_names=None,
                early_stop_patience=5, save_best_path="./best_model.pth"):
    
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    model.to(device)
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, total = 0.0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=train_correct / total)

        train_acc = train_correct / total
        writer.add_scalar("Loss/Train", train_loss / total, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)

        # ---------- Validation ----------
        model.eval()
        val_loss, val_correct = 0.0, 0
        total = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        loop_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for imgs, labels in loop_val:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                total += labels.size(0)

                for i in range(len(labels)):
                    class_total[labels[i].item()] += 1
                    if preds[i] == labels[i]:
                        class_correct[labels[i].item()] += 1

                loop_val.set_postfix(val_loss=loss.item(), acc=val_correct / total)

        val_acc = val_correct / total
        writer.add_scalar("Loss/Val", val_loss / total, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        if class_names:
            for class_id, class_name in enumerate(class_names):
                acc = class_correct[class_id] / class_total[class_id] if class_total[class_id] else 0.0
                writer.add_scalar(f"ClassAcc/{class_name}", acc, epoch)

        if scheduler:
            scheduler.step()

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # ---------- Early Stopping + Best 저장 ----------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_best_path)
            patience_counter = 0
            print(f"✅ Best model updated at epoch {epoch+1} (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1}. Best Val Acc: {best_val_acc:.4f}")
                break

    writer.close()
    model.load_state_dict(torch.load(save_best_path))
    return model

def train_model_amp(model, train_loader, val_loader, criterion, optimizer, scheduler=None,
                    num_epochs=10, device='cuda', log_dir="./runs/rock_classify_amp",
                    class_names=None, early_stop_patience=5, save_best_path="./best_model_amp.pth"):

    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    model.to(device)
    scaler = amp.GradScaler(device)
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, total = 0.0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with amp.autocast(device_type=device):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=train_correct / total)

        train_acc = train_correct / total
        writer.add_scalar("Loss/Train", train_loss / total, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)

        # ---------- Validation ----------
        model.eval()
        val_loss, val_correct = 0.0, 0
        total = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        loop_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for imgs, labels in loop_val:
                imgs, labels = imgs.to(device), labels.to(device)
                with amp.autocast(device_type=device):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                total += labels.size(0)

                for i in range(len(labels)):
                    class_total[labels[i].item()] += 1
                    if preds[i] == labels[i]:
                        class_correct[labels[i].item()] += 1

                loop_val.set_postfix(val_loss=loss.item(), acc=val_correct / total)

        val_acc = val_correct / total
        writer.add_scalar("Loss/Val", val_loss / total, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        if class_names:
            for class_id, class_name in enumerate(class_names):
                acc = class_correct[class_id] / class_total[class_id] if class_total[class_id] else 0.0
                writer.add_scalar(f"ClassAcc/{class_name}", acc, epoch)

        if scheduler:
            scheduler.step()

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # ---------- Early Stopping + Best 저장 ----------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_best_path)
            patience_counter = 0
            print(f"✅ Best model updated at epoch {epoch+1} (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1}. Best Val Acc: {best_val_acc:.4f}")
                break

    writer.close()
    model.load_state_dict(torch.load(save_best_path))
    return model