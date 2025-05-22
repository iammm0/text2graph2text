import torch
import torch.nn.functional as F

def nt_xent_loss(z_i, z_j, temperature=0.5):
    # Check if inputs are valid
    if z_i.shape[0] != z_j.shape[0] or z_i.shape[0] == 0:
        raise ValueError(f"[NT-Xent Loss] Invalid input: z_i.shape={z_i.shape}, z_j.shape={z_j.shape}")

    batch_size = z_i.shape[0]

    # L2 normalize
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    # Compute cosine similarity matrix
    similarity_matrix = torch.mm(z_i, z_j.T)  # shape: (batch_size, batch_size)

    # Ground-truth: each i should match j (diagonal)
    targets = torch.arange(batch_size).to(z_i.device)

    # Apply temperature scaling
    logits = similarity_matrix / temperature

    # CrossEntropy across rows and columns
    loss_i = F.cross_entropy(logits, targets)
    loss_j = F.cross_entropy(logits.T, targets)

    # Mean of forward + backward loss
    loss = (loss_i + loss_j) / 2

    return loss
