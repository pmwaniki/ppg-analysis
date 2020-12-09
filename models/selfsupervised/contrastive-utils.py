from pytorch_metric_learning.utils import accuracy_calculator
import numpy as np
import torch
import torch.nn as nn
import tqdm


class YourCalculator(accuracy_calculator.AccuracyCalculator):
    def calculate_precision_at_2(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(knn_labels, query_labels[:, None], 2,avg_of_avgs=True)

    def calculate_knn_accuracy(self, query_labels, cluster_labels, **kwargs):
        return 2.0

    def requires_clustering(self):
        return super().requires_clustering() + ["fancy_mutual_info"]

    def requires_knn(self):
        return super().requires_knn() + ["precision_at_2","knn_accuracy"]

def test_accuracy(model,encoder_test_loader):
    model.eval()
    test_loss = 0
    device=model.device
    with torch.no_grad():
        for x1_raw, x2_raw in tqdm(encoder_test_loader):
            x1_raw = x1_raw.to(device, dtype=torch.float)
            xis = model(x1_raw)
            x2_raw = x2_raw.to(device, dtype=torch.float)
            xjs = model(x2_raw)
            xis = nn.functional.normalize(xis, dim=1)
            xjs = nn.functional.normalize(xjs, dim=1)
            embeddings = torch.cat([xis, xjs], dim=0)
            labels = torch.cat([torch.arange(xis.shape[0]), ] * 2)
            loss = criterion(embeddings, labels)
            l1_reg = enc_l1_ * torch.norm(embeddings, p=1, dim=1).mean()
            loss += l1_reg

            test_loss += loss.item() / len(encoder_test_loader)
            # metrics = calculator.get_accuracy(xis.cpu().detach().numpy(), xjs.cpu().detach().numpy(),
            #                                   np.arange(xis.shape[0]), np.arange(xis.shape[0]),
            #                                   False)


if __name__=="__main__":
    cal=YourCalculator(k=1)
    cal0=accuracy_calculator.AccuracyCalculator(k=1)
    query=np.random.randn(30).reshape((5,6)).astype(np.float32)
    reference=query
    labs_query=np.array([0,1,0,0,2])
    labs_reference=np.array([0,1,0,0,2])
    cal.get_accuracy(query=query,reference=reference,query_labels=labs_query,reference_labels=labs_reference,
                     embeddings_come_from_same_source=False)