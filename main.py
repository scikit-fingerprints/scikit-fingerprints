from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.loader import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from rdkit.Chem import AllChem


if __name__ == "__main__":
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root="./datasets")

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(
        dataset[split_idx["train"]], batch_size=32, shuffle=True
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]], batch_size=32, shuffle=False
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]], batch_size=32, shuffle=False
    )

    print(train_loader.dataset[0].x)

    atom_enc = AtomEncoder(emb_dim=100)
    atom_emb = atom_enc(dataset.data.x)
    # aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    # Check if black's hook work
    print(atom_emb.shape)
