import torch
from sklearn.metrics import accuracy_score, f1_score
import time


def train_model(model, train_loader, val_loader, epochs, device):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_f1 = 0

    for epoch in range(epochs):

        start = time.time()

        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        preds = []
        true = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                out = model(x)
                p = torch.argmax(out, 1).cpu().numpy()

                preds.extend(p)
                true.extend(y.numpy())

        acc = accuracy_score(true, preds)
        f1 = f1_score(true, preds, average="macro")

        print(f"Epoch {epoch+1} | Loss {total_loss:.2f} | Val Acc {acc:.4f} | Val F1 {f1:.4f} | Time {time.time()-start:.1f}s")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_model.pth")

    print("Best F1:", best_f1)