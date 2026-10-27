"""Fast non-training checks for lunar classifier contracts and QCNN gradients."""
import torch
import tempfile
import copy
from pathlib import Path
from train import CLASS_TO_ID, CNN, ClassicalHead, HybridHead, LunarImages, pilot_indices
from train import fit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def main():
    root = __import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "raw" / "moon" / "lrocnet"
    train = LunarImages(root, "train")
    val = LunarImages(root, "val")
    assert CLASS_TO_ID == {"ejecta": 0, "oldcrater": 1, "none": 2}
    assert len(train) == 3608 and len(val) == 613
    assert len(pilot_indices(train, 1)) == 3 and len(pilot_indices(val, 1)) == 3
    cnn = CNN()
    output = cnn(torch.rand(2, 1, 227, 227))
    assert output.shape == (2, 3) and torch.isfinite(output).all()
    output.sum().backward()
    assert cnn.encoder.features[0].weight.grad is not None
    hybrid = HybridHead()
    logits = hybrid(torch.rand(2, 64))
    assert logits.shape == (2, 3) and torch.isfinite(logits).all()
    logits.sum().backward()
    grad_norm = sum(float(p.grad.norm()) for p in hybrid.parameters() if p.grad is not None)
    assert grad_norm > 0 and torch.isfinite(torch.tensor(grad_norm))
    control = ClassicalHead()
    assert control(torch.rand(2, 64)).shape == (2, 3)
    # Exercise epoch checkpoint save/resume with a tiny synthetic tensor set; no dataset or run output touched.
    x=torch.randn(12,4); y=torch.randint(0,3,(12,)); dl=DataLoader(TensorDataset(x,y),batch_size=4)
    torch.manual_seed(901); initial=nn.Linear(4,3); loss=nn.CrossEntropyLoss()
    continuous=copy.deepcopy(initial); opt_cont=torch.optim.Adam(continuous.parameters(),lr=0.01)
    torch.manual_seed(77); fit(continuous,dl,dl,opt_cont,loss,torch.device("cpu"),2,4)
    with tempfile.TemporaryDirectory() as temp:
        checkpoint=Path(temp)/"stage.pt"
        segmented=copy.deepcopy(initial); opt=torch.optim.Adam(segmented.parameters(),lr=0.01)
        torch.manual_seed(77)
        first=fit(segmented,dl,dl,opt,loss,torch.device("cpu"),1,4,checkpoint_path=checkpoint)
        resumed=copy.deepcopy(initial); opt2=torch.optim.Adam(resumed.parameters(),lr=0.01)
        combined=fit(resumed,dl,dl,opt2,loss,torch.device("cpu"),2,4,checkpoint_path=checkpoint,resume=True)
        assert len(first)==1 and len(combined)==2 and checkpoint.is_file()
        assert all(torch.equal(continuous.state_dict()[k],resumed.state_dict()[k]) for k in continuous.state_dict())
    print(f"PASS dataset counts/classes; CNN fwd+bwd; QCNN fwd+bwd grad_norm={grad_norm:.6g}; matched control; epoch checkpoint/resume")


if __name__ == "__main__":
    main()
