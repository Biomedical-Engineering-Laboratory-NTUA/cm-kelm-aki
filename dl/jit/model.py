import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input tensor to help the model understand the order of the sequence.
    The Transformer architecture does not have an inherent notion of position, so this class encodes
    positional information using sine and cosine functions of different frequencies.
    Args:
        d_model (int): Dimension of the model (the size of the input embedding).
        device (str): Device on which the positional encodings are calculated (e.g., 'cpu', 'cuda').
        max_len (int): Maximum length of the input sequence to encode positions for.
    """

    def __init__(self, d_model, device="cpu", max_len=500):
        super().__init__()

        # Create a positional encoding matrix of shape (max_len, d_model)
        self.pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).type(torch.float32)
        tmp = torch.arange(0, d_model, 2)
        den = 1 / torch.pow(torch.ones(int(d_model / 2)) * max_len, 2 * tmp / d_model)
        den = den.unsqueeze(0)

        # Populate the positional encoding matrix with sine and cosine functions
        self.pe[:, 0::2] = torch.sin(torch.matmul(pos, den))
        self.pe[:, 1::2] = torch.cos(torch.matmul(pos, den))
        self.pe = self.pe.to(device)

    def forward(self, x):
        """
        Adds the positional encoding to the input tensor.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            Tensor: Input tensor with positional encoding added.
        """
        return x + self.pe[: x.shape[1], :]


class PreLayer(nn.Module):
    """
    A pre-processing layer that applies a linear transformation to input data, mapping it to the model's
    dimensionality.
    Args:
        hid (int): Hidden layer size.
        d_model (int): Dimension of the model.
        drop_out (float): Dropout rate.
        in_dim (int): Dimension of the input data.
    """

    def __init__(self, hid, d_model, drop_out=0.0, in_dim=1):
        super().__init__()
        self.linear = nn.Linear(in_dim, d_model)

    def forward(self, x):
        """
        Transforms the input using a linear layer.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, in_dim).
        Returns:
            Tensor: Transformed tensor of shape (batch_size, seq_len, d_model).
        """

        out = self.linear(x)
        return out


class PostLayer(nn.Module):
    """
    A post-processing layer that applies a linear transformation to reduce the dimensionality to the desired
    output size.
    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hid (int): Hidden layer size.
        drop_out (float): Dropout rate.
    """

    def __init__(self, in_dim, out_dim, hid, drop_out=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        """
        Applies a linear transformation to the input data.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, in_dim).
        Returns:
            Tensor: Transformed tensor of shape (batch_size, seq_len, out_dim).
        """

        out = self.linear(x)
        return out


class SelfAttention(nn.Module):
    """
    Self-Attention mechanism to compute attention scores and apply them to the input sequence. This helps
    the model focus on different parts of the input sequence when making predictions.
    Args:
        d_model (int): Dimension of the model.
        n_head (int): Number of attention heads (for multi-head attention).
        attn_type (str): Type of attention mechanism (e.g., 'full' or 'causal').
    """

    def __init__(self, d_model, n_head, attn_type):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.n_head = n_head
        self.attn_type = attn_type
        self.softmax = nn.Softmax(dim=-1)
        scale = torch.sqrt(torch.FloatTensor([d_model // n_head]))
        self.register_buffer("scale", scale)

    def forward(self, x, mask=None, kv=None):
        """
        Applies the (self- or cross-) attention mechanism to the input sequence.
        Args:
            x (Tensor): Query input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor, optional): Mask to avoid attending to certain positions (e.g., padding or future tokens).
            kv (Tensor, optional): Key/value source of shape (batch_size, kv_len, d_model). When None the
                module performs standard self-attention (kv = x); when provided it performs cross-attention,
                with queries from ``x`` and keys/values from ``kv`` (e.g. the encoder memory).
        Returns:
            Tensor: Output of the attention mechanism with the same shape as ``x``.
        """

        if kv is None:
            kv = x
        B, N, D = x.shape
        M = kv.shape[1]
        q = self.query(x).view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        k = self.key(kv).view(B, M, self.n_head, D // self.n_head).transpose(1, 2)
        v = self.value(kv).view(B, M, self.n_head, D // self.n_head).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.softmax(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        return out


class FeedForward(nn.Module):
    """
    A two-layer feed-forward neural network applied independently to each position in the sequence.
    Args:
        d_model (int): Dimension of the model.
        ff_hidnum (int): Size of the hidden layer in the feed-forward network.
    """

    def __init__(self, d_model, ff_hidnum):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ff_hidnum)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ff_hidnum, d_model)

    def forward(self, x):
        """
        Applies the feed-forward network to the input.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            Tensor: Output tensor with the same shape as the input.
        """
        return self.linear2(self.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    """
    A single layer of the Transformer encoder. It consists of a self-attention mechanism followed by
    a feed-forward network, both with residual connections and layer normalization.
    Args:
        d_model (int): Dimension of the model.
        n_head (int): Number of attention heads.
        attn_type (str): Type of attention.
        ff_hidnum (int): Hidden layer size of the feed-forward network.
        drop_out (float): Dropout rate.
    """

    def __init__(self, d_model, n_head, attn_type, ff_hidnum, drop_out=0.1):
        super().__init__()
        self.attn = SelfAttention(d_model, n_head, attn_type)
        self.ff = FeedForward(d_model, ff_hidnum)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)

    def forward(self, x, mask=None):
        """
        Forward pass through a single encoder layer.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor, optional): Optional attention mask.
        Returns:
            Tensor: Output of the encoder layer with the same shape as the input.
        """

        x = self.norm1(x + self.dropout1(self.attn(x, mask)))
        x = self.norm2(x + self.dropout2(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    """
    A single layer of the Transformer decoder. It consists of two attention mechanisms (self-attention
    and encoder-decoder attention) followed by a feed-forward network. Residual connections and
    layer normalization are applied after each sub-layer.
    Args:
        d_model (int): Dimension of the model.
        n_head (int): Number of attention heads.
        attn_type (str): Type of attention.
        ff_hidnum (int): Hidden layer size of the feed-forward network.
        drop_out (float): Dropout rate.
    """

    def __init__(self, d_model, n_head, attn_type, ff_hidnum, drop_out=0.1):
        super().__init__()
        self.attn1 = SelfAttention(d_model, n_head, attn_type)
        self.attn2 = SelfAttention(d_model, n_head, attn_type)
        self.ff = FeedForward(d_model, ff_hidnum)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)
        self.dropout3 = nn.Dropout(drop_out)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        """
        Forward pass through a single decoder layer.
        Args:
            x (Tensor): Target sequence input.
            memory (Tensor): Output from the encoder.
            src_mask (Tensor): Mask for the source sequence.
            tgt_mask (Tensor): Mask for the target sequence.
        Returns:
            Tensor: Output of the decoder layer with the same shape as the input.
        """

        x = self.norm1(x + self.dropout1(self.attn1(x, mask=tgt_mask)))
        x = self.norm2(x + self.dropout2(self.attn2(x, mask=src_mask, kv=memory)))
        x = self.norm3(x + self.dropout3(self.ff(x)))
        return x


class Encoder(nn.Module):
    """
    The Transformer encoder, which is composed of multiple stacked encoder layers.
    Args:
        N (int): Number of encoder layers.
        d_model (int): Dimension of the model.
        n_head (int): Number of attention heads.
        attn_type (str): Type of attention.
        ff_hidnum (int): Hidden layer size of the feed-forward network.
        drop_out (float): Dropout rate.
    """

    def __init__(self, N, d_model, n_head, attn_type, ff_hidnum, drop_out=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, n_head, attn_type, ff_hidnum, drop_out)
                for _ in range(N)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Forward pass through the encoder.
        Args:
            x (Tensor): Input sequence tensor.
        Returns:
            Tensor: Final output of the encoder.
        """

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """
    The Transformer decoder, which is composed of multiple stacked decoder layers.
    Args:
        N (int): Number of decoder layers.
        d_model (int): Dimension of the model.
        n_head (int): Number of attention heads.
        attn_type (str): Type of attention.
        ff_hidnum (int): Hidden layer size of the feed-forward network.
        drop_out (float): Dropout rate.
    """

    def __init__(self, N, d_model, n_head, attn_type, ff_hidnum, drop_out=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, n_head, attn_type, ff_hidnum, drop_out)
                for _ in range(N)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        """
        Forward pass through the decoder.
        Args:
            x (Tensor): Target sequence tensor.
            memory (Tensor): Encoder output (source sequence representation).
            src_mask (Tensor): Mask for the source sequence.
            tgt_mask (Tensor): Mask for the target sequence.
        Returns:
            Tensor: Final output of the decoder.
        """

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)



class TabularTokenizer(nn.Module):
    """
    Turns a row of tabular features into a sequence of d_model tokens, one per feature.
    Categorical features use a per-column embedding table (as in TabTransformer); continuous
    features use a per-feature affine map x_j * W_j + b_j (as in FT-Transformer). Since every
    column has its own parameters, tokens already carry their column identity and no
    positional encoding is added (tabular features have no order).
    Args:
        cat_cardinalities (list[int]): Embedding table size of each categorical column (index 0 = unknown).
        num_continuous (int): Number of continuous columns.
        d_model (int): Dimension of the model.
    """

    def __init__(self, cat_cardinalities, num_continuous, d_model):
        super().__init__()
        self.num_categorical = len(cat_cardinalities)
        self.num_continuous = num_continuous
        self.cat_emb = nn.ModuleList(
            [nn.Embedding(c, d_model) for c in cat_cardinalities]
        )
        if num_continuous > 0:
            bound = 1 / d_model**0.5
            self.cont_weight = nn.Parameter(
                torch.empty(num_continuous, d_model).uniform_(-bound, bound)
            )
            self.cont_bias = nn.Parameter(
                torch.empty(num_continuous, d_model).uniform_(-bound, bound)
            )

    def forward(self, x):
        """
        Args:
            x (Tensor): Packed rows of shape (B, num_categorical + num_continuous); the first
                num_categorical columns hold category indices stored as floats.
        Returns:
            Tensor: Feature tokens of shape (B, num_categorical + num_continuous, d_model).
        """

        tokens = []
        if self.num_categorical > 0:
            x_cat = x[:, : self.num_categorical].long()
            tokens.append(
                torch.stack(
                    [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_emb)], dim=1
                )
            )
        if self.num_continuous > 0:
            x_cont = x[:, self.num_categorical :]
            tokens.append(x_cont.unsqueeze(-1) * self.cont_weight + self.cont_bias)
        return torch.cat(tokens, dim=1)


class TabularTransformer(nn.Module):
    """
    JITrans encoder-decoder for tabular classification, preserving the original autoregressive
    (feedback) form.

    Each row is turned by ``TabularTokenizer`` into one token per feature, which the encoder
    processes into a memory. The decoder then produces the class prediction autoregressively:
    starting from a single learnable prediction token it runs ``feedback_steps`` refinement
    steps, appending the softmax of each step's logits back as the next decoder token (the
    "retroazione"/feedback). A causal mask keeps the decoder self-attention autoregressive,
    while the decoder cross-attends to the encoder memory.

    Args:
        device (str): Device on which the model will be run ('cpu' or 'cuda').
        cat_cardinalities (list[int]): Embedding table size of each categorical column.
        num_continuous (int): Number of continuous columns.
        d_model (int): Dimension of the model.
        attn_type (str): Type of attention mechanism.
        N_enc (int): Number of encoder layers.
        N_dec (int): Number of decoder layers.
        h_enc (int): Number of attention heads in the encoder.
        h_dec (int): Number of attention heads in the decoder.
        ff_hidnum (int): Hidden layer size of the feed-forward network.
        dropout_model (float): Dropout rate for the encoder and decoder.
        num_classes (int): Number of output classes.
        feedback_steps (int): Number of autoregressive feedback/refinement steps performed by the decoder.
        use_checkpoint (bool): Whether to use gradient checkpointing to save memory.
    """

    def __init__(
        self,
        device,
        cat_cardinalities,
        num_continuous,
        d_model,
        attn_type,
        N_enc,
        N_dec,
        h_enc,
        h_dec,
        ff_hidnum,
        dropout_model,
        num_classes,
        feedback_steps=4,
        use_checkpoint=False,
    ):
        super().__init__()
        self.device = device
        self.use_checkpoint = use_checkpoint
        self.num_classes = num_classes
        self.feedback_steps = feedback_steps
        self.y_pre = PreLayer(None, d_model, 0.0, num_classes)
        self.pos = PositionalEncoding(d_model, device=device)
        self.enc = Encoder(N_enc, d_model, h_enc, attn_type, ff_hidnum, dropout_model)
        self.dec = Decoder(N_dec, d_model, h_dec, attn_type, ff_hidnum, dropout_model)
        self.post = PostLayer(d_model, num_classes, None, 0.0)
        # Learnable start token of the decoder, expressed in class-distribution space.
        self.y_start = nn.Parameter(torch.zeros(1, 1, num_classes))
        self.tokenizer = TabularTokenizer(cat_cardinalities, num_continuous, d_model)

    def _causal_mask(self, n, device):
        """Lower-triangular causal mask broadcastable to (1, 1, n, n)."""
        mask = torch.tril(torch.ones(n, n, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x, return_all_steps=False):
        """
        Forward pass: encode the feature tokens, then autoregressively refine the class prediction.
        Args:
            x (Tensor): Packed rows of shape (B, num_categorical + num_continuous).
            return_all_steps (bool): If True, also return the per-step logits stack (B, feedback_steps, num_classes).
        Returns:
            Tensor: Final-step class logits of shape (B, num_classes). If ``return_all_steps`` is True,
                returns a tuple ``(final_logits, all_step_logits)``.
        """

        # --- Encoder: feature tokens -> memory ---
        x_emb = self.tokenizer(x)
        if self.use_checkpoint and self.training:
            memory = checkpoint(self.enc, x_emb, use_reentrant=True)
        else:
            memory = self.enc(x_emb)

        # --- Decoder: autoregressive feedback refinement ---
        B = x.shape[0]
        y = self.y_start.expand(B, -1, -1)  # (B, 1, num_classes)
        step_logits_list = []
        for _ in range(self.feedback_steps):
            n = y.shape[1]
            y_emb_pos = self.pos(self.y_pre(y))
            tgt_mask = self._causal_mask(n, x.device)
            out = self.dec(y_emb_pos, memory, src_mask=None, tgt_mask=tgt_mask)
            step_logits = self.post(out)[:, -1]  # (B, num_classes) for the latest token
            step_logits_list.append(step_logits)
            # Feedback: append the predicted class distribution as the next decoder token.
            next_token = torch.softmax(step_logits, dim=-1).unsqueeze(
                1
            )  # (B, 1, num_classes)
            y = torch.cat([y, next_token], dim=1)

        final_logits = step_logits_list[-1]
        if return_all_steps:
            return final_logits, torch.stack(step_logits_list, dim=1)
        return final_logits


def define_model(
    device,
    cat_cardinalities,
    num_continuous,
    num_classes,
    d_model=32,
    n_heads=8,
    N_enc=3,
    N_dec=2,
    dropout=0.1,
    feedback_steps=4,
    use_checkpoint=False,
):
    """
    Utility function to create the JITrans tabular classifier.
    Args:
        device (str): Device on which the model will be run ('cpu' or 'cuda').
        cat_cardinalities (list[int]): Embedding table size of each categorical column.
        num_continuous (int): Number of continuous columns.
        num_classes (int): Number of output classes.
        d_model (int): Dimension of the model (must be even and divisible by n_heads).
        n_heads (int): Number of attention heads in encoder and decoder.
        N_enc (int): Number of encoder layers.
        N_dec (int): Number of decoder layers.
        dropout (float): Dropout rate of the encoder and decoder.
        feedback_steps (int): Number of autoregressive feedback/refinement steps.
        use_checkpoint (bool): Whether to use gradient checkpointing to save memory during training.
    Returns:
        TabularTransformer: The instantiated model.
    """

    if d_model % n_heads != 0 or d_model % 2 != 0:
        raise ValueError(
            f"d_model ({d_model}) must be even and divisible by n_heads ({n_heads})."
        )
    if len(cat_cardinalities) + num_continuous == 0:
        raise ValueError("The tabular model needs at least one feature.")

    model = TabularTransformer(
        device=device,
        cat_cardinalities=cat_cardinalities,
        num_continuous=num_continuous,
        d_model=d_model,
        attn_type="full",
        N_enc=N_enc,
        N_dec=N_dec,
        h_enc=n_heads,
        h_dec=n_heads,
        ff_hidnum=4 * d_model,
        dropout_model=dropout,
        num_classes=num_classes,
        feedback_steps=feedback_steps,
        use_checkpoint=use_checkpoint,
    ).to(device)

    return model
