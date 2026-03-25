import torch
import pandas as pd


def predict(model, test_loader, device, inv_map, test_df):

    model.eval()
    preds = []

    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            out = model(x)
            p = torch.argmax(out, 1).cpu().numpy()
            preds.extend(p)

    submission = pd.DataFrame({
        "id": test_df["id"],
        "genre": [inv_map[p] for p in preds]
    })

    submission.to_csv("submission.csv", index=False)
    print("Submission saved!")