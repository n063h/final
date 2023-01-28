


from torchmetrics import MetricCollection,Accuracy,F1Score,Recall,Precision


def get_multiclass_metrics_collections(num_classes,device):
    return MetricCollection({
            'acc':Accuracy(task="multiclass", num_classes=num_classes),'f1':F1Score(task="multiclass", num_classes=num_classes),
            'rec':Recall(task="multiclass", num_classes=num_classes),'pr':Precision(task="multiclass", num_classes=num_classes)
        }).to(device)
    
def get_multiclass_acc_metrics(num_classes,device):
    return MetricCollection({'acc':Accuracy(task="multiclass", num_classes=num_classes)}).to(device)